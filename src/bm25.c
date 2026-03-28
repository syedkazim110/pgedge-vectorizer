/*-------------------------------------------------------------------------
 *
 * bm25.c
 *		BM25 sparse vector computation for hybrid search support
 *
 * Implements tokenization, IDF statistics management, and BM25 scoring
 * to produce sparsevec values stored alongside dense embeddings in chunk
 * tables.  Functions that touch the database assume the caller holds an
 * active SPI connection (except the SQL-callable wrappers, which open
 * their own SPI session).
 *
 * Copyright (c) 2025 - 2026, pgEdge, Inc.
 *
 *-------------------------------------------------------------------------
 */
#include "bm25.h"

#include <ctype.h>
#include <math.h>
#include <string.h>

#include "access/xact.h"
#include "lib/stringinfo.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/memutils.h"

/* ----------------------------------------------------------------
 * English stopword list — must stay sorted (used with bsearch)
 * ----------------------------------------------------------------
 */
static const char *stopwords[] = {
	"a", "above", "after", "again", "an", "and", "are", "as",
	"at", "be", "been", "before", "being", "below", "both",
	"but", "by", "can", "could", "dare", "did", "do", "does",
	"down", "during", "either", "for", "from", "further",
	"had", "has", "have", "if", "in", "into", "is", "may",
	"might", "need", "neither", "nor", "not", "of", "off",
	"on", "once", "or", "ought", "out", "over", "shall",
	"should", "so", "the", "then", "through", "to", "under",
	"up", "used", "was", "were", "whether", "while", "will",
	"with", "would", "yet"
};

#define STOPWORDS_COUNT \
	((int) (sizeof(stopwords) / sizeof(stopwords[0])))

/* Comparator for bsearch against the stopwords array */
static int
stopword_cmp(const void *a, const void *b)
{
	return strcmp((const char *) a, *(const char * const *) b);
}

/*
 * is_stopword — true if term appears in the sorted stopword list
 */
static bool
is_stopword(const char *term)
{
	return bsearch(term, stopwords, STOPWORDS_COUNT,
				   sizeof(stopwords[0]), stopword_cmp) != NULL;
}

/*
 * djb2_hash — fast string hash for mapping terms to dimension indices
 */
static uint32
djb2_hash(const char *str)
{
	uint32				hash = 5381;
	const unsigned char *p = (const unsigned char *) str;

	while (*p)
		hash = ((hash << 5) + hash) ^ *p++;

	return hash;
}

/*
 * term_to_dim — map a term to a 1-based sparse vector dimension
 * in [1, BM25_VOCAB_SIZE]
 */
static int
term_to_dim(const char *term)
{
	return (int) (djb2_hash(term) % (uint32) BM25_VOCAB_SIZE) + 1;
}

/*
 * normalize_text_inplace — lowercase and replace non-alpha with spaces
 */
static void
normalize_text_inplace(char *buf)
{
	char *p;

	for (p = buf; *p; p++)
	{
		if (isalpha((unsigned char) *p))
			*p = (char) tolower((unsigned char) *p);
		else
			*p = ' ';
	}
}

/*
 * find_term_idx — return index of tok in terms[], or -1 if not found
 */
static int
find_term_idx(BM25Term *terms, int nterms, const char *tok)
{
	int i;

	for (i = 0; i < nterms; i++)
	{
		if (strcmp(terms[i].term, tok) == 0)
			return i;
	}
	return -1;
}

/*
 * grow_terms — double the capacity of the terms array
 */
static BM25Term *
grow_terms(BM25Term *terms, int *cap)
{
	*cap = (*cap == 0) ? 16 : *cap * 2;
	if (terms == NULL)
		return palloc(*cap * sizeof(BM25Term));
	return repalloc(terms, *cap * sizeof(BM25Term));
}

/*
 * bm25_tokenize
 *
 * Lowercase the input, replace non-alpha chars with spaces, split on
 * whitespace, strip English stopwords, deduplicate, and count term
 * frequencies.  Returns a palloc'd BM25Term array; *ntokens is the
 * number of distinct terms.  Returns an empty array (never NULL) for
 * NULL or empty input.
 */
BM25Term *
bm25_tokenize(const char *text, int *ntokens)
{
	char       *buf;
	char       *tok;
	BM25Term   *terms = NULL;
	int         nterms = 0;
	int         cap = 0;
	int         idx;

	*ntokens = 0;

	if (text == NULL || *text == '\0')
		return palloc0(sizeof(BM25Term));

	buf = pstrdup(text);
	normalize_text_inplace(buf);

	tok = strtok(buf, " ");
	while (tok != NULL)
	{
		if (*tok != '\0' && !is_stopword(tok))
		{
			idx = find_term_idx(terms, nterms, tok);
			if (idx >= 0)
			{
				terms[idx].tf++;
			}
			else
			{
				if (nterms >= cap)
					terms = grow_terms(terms, &cap);
				terms[nterms].term = pstrdup(tok);
				terms[nterms].tf   = 1;
				nterms++;
			}
		}
		tok = strtok(NULL, " ");
	}

	pfree(buf);

	if (terms == NULL)
		terms = palloc0(sizeof(BM25Term));

	*ntokens = nterms;
	return terms;
}

/*
 * bm25_load_idf_stats
 *
 * Load all rows from {chunk_table}_idf_stats via SPI.  Returns a
 * palloc'd IdfStat array; *nstats is set to the row count.  Returns
 * NULL (not an error) if the stats table does not exist yet.
 *
 * Uses a subtransaction so a missing table does not abort the outer
 * worker transaction.  Caller must have an active SPI connection.
 */

/*
 * read_idf_row — extract one IdfStat from a SPI result row
 */
static IdfStat
read_idf_row(SPITupleTable *tuptable, int i)
{
	IdfStat  s;
	bool     isnull;
	Datum    val;

	val = SPI_getbinval(tuptable->vals[i], tuptable->tupdesc, 1, &isnull);
	s.term = isnull ? pstrdup("") : pstrdup(TextDatumGetCString(val));

	val = SPI_getbinval(tuptable->vals[i], tuptable->tupdesc, 2, &isnull);
	s.doc_freq = isnull ? 1 : DatumGetInt32(val);

	val = SPI_getbinval(tuptable->vals[i], tuptable->tupdesc, 3, &isnull);
	s.total_docs = isnull ? 1 : DatumGetInt32(val);

	val = SPI_getbinval(tuptable->vals[i], tuptable->tupdesc, 4, &isnull);
	s.idf_weight = isnull ? 0.693 : DatumGetFloat8(val);

	return s;
}

IdfStat *
bm25_load_idf_stats(const char *chunk_table, int *nstats)
{
	char            *sql;
	int              ret;
	IdfStat         *stats = NULL;
	int              n;
	bool             ok = false;
	MemoryContext    oldctx;
	ResourceOwner    oldowner;

	*nstats = 0;

	sql = psprintf(
		"SELECT term, doc_freq, total_docs, idf_weight "
		"FROM %s_idf_stats",
		chunk_table);

	oldctx   = CurrentMemoryContext;
	oldowner = CurrentResourceOwner;

	/*
	 * Wrap in a subtransaction so that "table does not exist" is
	 * handled gracefully rather than aborting the worker's batch.
	 */
	BeginInternalSubTransaction(NULL);
	PG_TRY();
	{
		ret = SPI_execute(sql, true, 0);
		ok  = true;
		ReleaseCurrentSubTransaction();
	}
	PG_CATCH();
	{
		MemoryContextSwitchTo(oldctx);
		CurrentResourceOwner = oldowner;
		RollbackAndReleaseCurrentSubTransaction();
		FlushErrorState();
		pfree(sql);
		return NULL;
	}
	PG_END_TRY();

	pfree(sql);

	if (!ok || ret != SPI_OK_SELECT || SPI_processed == 0)
		return NULL;

	n     = (int) SPI_processed;
	stats = palloc(n * sizeof(IdfStat));

	for (int i = 0; i < n; i++)
		stats[i] = read_idf_row(SPI_tuptable, i);

	*nstats = n;
	return stats;
}

/*
 * bm25_avg_doc_len_internal
 *
 * Return AVG(length(content)) from chunk_table via SPI.
 * Returns 1.0 if the table is empty.
 * Caller must have an active SPI connection.
 */
float8
bm25_avg_doc_len_internal(const char *chunk_table)
{
	char   *sql;
	int     ret;
	bool    isnull;
	Datum   val;
	float8  result = 1.0;

	sql = psprintf("SELECT AVG(length(content)) FROM %s",
				   chunk_table);
	ret = SPI_execute(sql, true, 1);
	pfree(sql);

	if (ret != SPI_OK_SELECT)
	{
		elog(WARNING,
			 "bm25_avg_doc_len_internal: SPI error for table %s",
			 chunk_table);
		return 1.0;
	}

	if (SPI_processed == 0)
		return 1.0;

	val = SPI_getbinval(SPI_tuptable->vals[0],
						SPI_tuptable->tupdesc, 1, &isnull);
	if (!isnull)
		result = DatumGetFloat8(val);

	return (result <= 0.0) ? 1.0 : result;
}

/*
 * lookup_idf — find the IDF weight for a term.
 * Returns ln(2) ≈ 0.693 (i.e. idf = ln(1+1)) when not found.
 */
static float8
lookup_idf(const char *term, IdfStat *stats, int nstats)
{
	for (int i = 0; i < nstats; i++)
	{
		if (strcmp(stats[i].term, term) == 0)
			return stats[i].idf_weight;
	}
	return log(2.0);
}

typedef struct { int dim; float8 score; } DimScore;

/*
 * accumulate_dim_score — add score to existing dim or append a new entry
 */
static void
accumulate_dim_score(DimScore **pairs, int *npairs, int *cap,
					 int dim, float8 score)
{
	for (int j = 0; j < *npairs; j++)
	{
		if ((*pairs)[j].dim == dim)
		{
			(*pairs)[j].score += score;
			return;
		}
	}

	if (*npairs >= *cap)
	{
		*cap = (*cap == 0) ? 16 : *cap * 2;
		if (*pairs == NULL)
			*pairs = palloc(*cap * sizeof(DimScore));
		else
			*pairs = repalloc(*pairs, *cap * sizeof(DimScore));
	}
	(*pairs)[*npairs].dim   = dim;
	(*pairs)[*npairs].score = score;
	(*npairs)++;
}

/*
 * build_sparsevec_string — format DimScore pairs as "{d:s,...}/N"
 */
static char *
build_sparsevec_string(DimScore *pairs, int npairs)
{
	StringInfoData  buf;
	bool            first = true;

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '{');
	for (int i = 0; i < npairs; i++)
	{
		if (!first)
			appendStringInfoChar(&buf, ',');
		appendStringInfo(&buf, "%d:%f", pairs[i].dim, pairs[i].score);
		first = false;
	}
	appendStringInfo(&buf, "}/%d", BM25_VOCAB_SIZE);
	return buf.data;
}

/*
 * bm25_compute_sparse_str
 *
 * Compute BM25 scores for every token and build a sparsevec string of
 * the form "{dim1:score1,dim2:score2}/VOCAB_SIZE".  Terms that hash to
 * the same dimension have their scores summed (collision resolution).
 * Only terms with score > 0 are included.
 *
 * Returns a palloc'd string; never returns NULL.
 */
char *
bm25_compute_sparse_str(BM25Term *tokens, int ntokens,
						IdfStat *stats, int nstats,
						float8 k1, float8 b,
						float8 avg_doc_len, int doc_len)
{
	DimScore   *pairs = NULL;
	int         npairs = 0;
	int         cap = 0;
	float8      dl_ratio;
	char       *result;

	if (avg_doc_len <= 0.0)
		avg_doc_len = 1.0;
	dl_ratio = (float8) doc_len / avg_doc_len;

	for (int i = 0; i < ntokens; i++)
	{
		float8  tf    = (float8) tokens[i].tf;
		float8  idf   = lookup_idf(tokens[i].term, stats, nstats);
		float8  denom = tf + k1 * (1.0 - b + b * dl_ratio);
		float8  score = (denom <= 0.0) ? 0.0 :
						idf * (tf * (k1 + 1.0)) / denom;

		if (score > 0.0)
			accumulate_dim_score(&pairs, &npairs, &cap,
								 term_to_dim(tokens[i].term), score);
	}

	result = build_sparsevec_string(pairs, npairs);

	if (pairs != NULL)
		pfree(pairs);

	return result;
}

/*
 * bm25_compute_sparse_vector
 *
 * Wraps bm25_compute_sparse_str and casts the string to a sparsevec
 * Datum via SPI.  Caller must have an active SPI connection.
 */
Datum
bm25_compute_sparse_vector(BM25Term *tokens, int ntokens,
						   IdfStat *stats,  int nstats,
						   float8 k1, float8 b,
						   float8 avg_doc_len, int doc_len)
{
	char   *vec_str;
	char   *sql;
	int     ret;
	Datum   result;
	bool    isnull;

	vec_str = bm25_compute_sparse_str(tokens, ntokens,
									  stats, nstats,
									  k1, b,
									  avg_doc_len, doc_len);

	sql = psprintf("SELECT '%s'::sparsevec", vec_str);
	ret = SPI_execute(sql, true, 1);
	pfree(sql);
	pfree(vec_str);

	if (ret != SPI_OK_SELECT || SPI_processed != 1)
		elog(ERROR,
			 "bm25_compute_sparse_vector: "
			 "failed to cast string to sparsevec");

	result = SPI_getbinval(SPI_tuptable->vals[0],
						   SPI_tuptable->tupdesc, 1, &isnull);

	if (isnull)
		elog(ERROR,
			 "bm25_compute_sparse_vector: NULL sparsevec result");

	return SPI_datumTransfer(result, false, -1);
}

/*
 * bm25_update_idf_stats
 *
 * Upsert each token's term into {chunk_table}_idf_stats, incrementing
 * doc_freq and recomputing idf_weight.  Runs inside a subtransaction
 * so that a failure does not abort the outer worker transaction.
 * Caller must have an active SPI connection.
 */
/*
 * upsert_term_idf — insert or update a single term's IDF stats row
 */
static void
upsert_term_idf(const char *chunk_table, const char *term)
{
	char   *safe_term = quote_literal_cstr(term);
	char   *sql;
	int     ret;

	sql = psprintf(
		"INSERT INTO %s_idf_stats"
		"    (term, doc_freq, total_docs, idf_weight)"
		" SELECT %s, 1, t.cnt,"
		"        ln(1.0 + (t.cnt - 1.0 + 0.5)"
		"               / (1.0 + 0.5))"
		" FROM (SELECT count(*)::int AS cnt"
		"       FROM %s) t"
		" ON CONFLICT (term) DO UPDATE SET"
		"    doc_freq   = %s_idf_stats.doc_freq + 1,"
		"    total_docs = EXCLUDED.total_docs,"
		"    idf_weight = ln(1.0 +"
		"        (EXCLUDED.total_docs"
		"         - (%s_idf_stats.doc_freq + 1) + 0.5)"
		"        / ((%s_idf_stats.doc_freq + 1) + 0.5)),"
		"    updated_at = now()",
		chunk_table, safe_term, chunk_table,
		chunk_table, chunk_table, chunk_table);

	pfree(safe_term);
	ret = SPI_execute(sql, false, 0);
	pfree(sql);

	if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE)
		elog(WARNING,
			 "bm25_update_idf_stats: unexpected SPI result %d "
			 "for term in table %s", ret, chunk_table);
}

void
bm25_update_idf_stats(const char *chunk_table,
					  BM25Term *tokens, int ntokens)
{
	MemoryContext oldctx;
	ResourceOwner oldowner;

	if (ntokens <= 0)
		return;

	oldctx   = CurrentMemoryContext;
	oldowner = CurrentResourceOwner;

	BeginInternalSubTransaction(NULL);
	PG_TRY();
	{
		for (int i = 0; i < ntokens; i++)
			upsert_term_idf(chunk_table, tokens[i].term);

		ReleaseCurrentSubTransaction();
	}
	PG_CATCH();
	{
		MemoryContextSwitchTo(oldctx);
		CurrentResourceOwner = oldowner;
		RollbackAndReleaseCurrentSubTransaction();
		FlushErrorState();
		elog(WARNING,
			 "bm25_update_idf_stats: subtransaction rolled back "
			 "for chunk table %s; IDF stats may be stale",
			 chunk_table);
	}
	PG_END_TRY();
}

/* ----------------------------------------------------------------
 * SQL-callable wrappers — PG_FUNCTION_INFO_V1 must appear in exactly
 * one translation unit (bm25.c); bm25.h uses plain extern declarations.
 * ----------------------------------------------------------------
 */
PG_FUNCTION_INFO_V1(pgedge_vectorizer_bm25_query_vector);
PG_FUNCTION_INFO_V1(pgedge_vectorizer_bm25_avg_doc_len);
PG_FUNCTION_INFO_V1(pgedge_vectorizer_bm25_tokenize);

/* ----------------------------------------------------------------
 * SQL-callable wrapper:
 *   pgedge_vectorizer.bm25_query_vector(query TEXT, chunk_table TEXT)
 *   RETURNS sparsevec
 * ----------------------------------------------------------------
 */
Datum
pgedge_vectorizer_bm25_query_vector(PG_FUNCTION_ARGS)
{
	text       *query_text;
	text       *chunk_table_text;
	const char *query;
	const char *chunk_table;
	BM25Term   *tokens;
	IdfStat    *stats;
	int         ntokens;
	int         nstats;
	float8      avg_doc_len;
	Datum       result;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		ereport(ERROR,
				(errmsg("bm25_query_vector: query and chunk_table "
						"must not be NULL")));

	query_text       = PG_GETARG_TEXT_PP(0);
	chunk_table_text = PG_GETARG_TEXT_PP(1);
	query            = TextDatumGetCString(
						   PointerGetDatum(query_text));
	chunk_table      = TextDatumGetCString(
						   PointerGetDatum(chunk_table_text));

	SPI_connect();

	tokens      = bm25_tokenize(query, &ntokens);
	stats       = bm25_load_idf_stats(chunk_table, &nstats);
	avg_doc_len = bm25_avg_doc_len_internal(chunk_table);

	result = bm25_compute_sparse_vector(
				 tokens, ntokens,
				 stats,  nstats,
				 pgedge_vectorizer_bm25_k1,
				 pgedge_vectorizer_bm25_b,
				 avg_doc_len,
				 (int) VARSIZE_ANY_EXHDR(query_text));

	SPI_finish();

	PG_RETURN_DATUM(result);
}

/* ----------------------------------------------------------------
 * SQL-callable wrapper:
 *   pgedge_vectorizer.bm25_avg_doc_len(chunk_table TEXT)
 *   RETURNS float8
 * ----------------------------------------------------------------
 */
Datum
pgedge_vectorizer_bm25_avg_doc_len(PG_FUNCTION_ARGS)
{
	text       *chunk_table_text;
	const char *chunk_table;
	float8      result;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errmsg("bm25_avg_doc_len: chunk_table must "
						"not be NULL")));

	chunk_table_text = PG_GETARG_TEXT_PP(0);
	chunk_table = TextDatumGetCString(
					  PointerGetDatum(chunk_table_text));

	SPI_connect();
	result = bm25_avg_doc_len_internal(chunk_table);
	SPI_finish();

	PG_RETURN_FLOAT8(result);
}

/* ----------------------------------------------------------------
 * SQL-callable wrapper:
 *   pgedge_vectorizer.bm25_tokenize(query TEXT) RETURNS TEXT[]
 *
 * Exposes the tokenizer for testing and debugging.
 * ----------------------------------------------------------------
 */
Datum
pgedge_vectorizer_bm25_tokenize(PG_FUNCTION_ARGS)
{
	text       *input;
	const char *text_str;
	BM25Term   *tokens;
	int         ntokens;
	ArrayType  *result;
	Datum      *elems;
	int         i;

	if (PG_ARGISNULL(0))
	{
		result = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(result);
	}

	input    = PG_GETARG_TEXT_PP(0);
	text_str = TextDatumGetCString(PointerGetDatum(input));

	tokens = bm25_tokenize(text_str, &ntokens);

	if (ntokens == 0)
	{
		result = construct_empty_array(TEXTOID);
		PG_RETURN_ARRAYTYPE_P(result);
	}

	elems = palloc(ntokens * sizeof(Datum));
	for (i = 0; i < ntokens; i++)
		elems[i] = CStringGetTextDatum(tokens[i].term);

	result = construct_array(elems, ntokens, TEXTOID,
							 -1, false, TYPALIGN_INT);

	PG_RETURN_ARRAYTYPE_P(result);
}
