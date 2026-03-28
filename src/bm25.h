/*-------------------------------------------------------------------------
 *
 * bm25.h
 *		BM25 sparse vector computation for hybrid search support
 *
 * Copyright (c) 2025 - 2026, pgEdge, Inc.
 *
 *-------------------------------------------------------------------------
 */
#ifndef PGEDGE_BM25_H
#define PGEDGE_BM25_H

#include "pgedge_vectorizer.h"

/*
 * Vocabulary size for hashing terms to sparse vector dimensions.
 * Large enough to keep collision rates low for typical corpora.
 */
#define BM25_VOCAB_SIZE  65536

/*
 * Term frequency entry produced by bm25_tokenize()
 */
typedef struct
{
	char   *term;
	int     tf;       /* term frequency in this document */
} BM25Term;

/*
 * IDF statistics for a term loaded from the _idf_stats table
 */
typedef struct
{
	char   *term;
	int     doc_freq;
	int     total_docs;
	float8  idf_weight;
} IdfStat;

/*
 * GUC variables defined in guc.c, extern-declared here so that
 * bm25.c and worker.c can read them after including this header.
 */
extern bool   pgedge_vectorizer_enable_hybrid;
extern double pgedge_vectorizer_bm25_k1;
extern double pgedge_vectorizer_bm25_b;

/*
 * Internal functions — callers must have an active SPI connection
 * for bm25_load_idf_stats, bm25_update_idf_stats, and
 * bm25_avg_doc_len_internal.
 */
BM25Term *bm25_tokenize(const char *text, int *ntokens);
IdfStat  *bm25_load_idf_stats(const char *chunk_table, int *nstats);
char     *bm25_compute_sparse_str(BM25Term *tokens, int ntokens,
								  IdfStat *stats, int nstats,
								  float8 k1, float8 b,
								  float8 avg_doc_len, int doc_len);
Datum     bm25_compute_sparse_vector(BM25Term *tokens, int ntokens,
									 IdfStat *stats,  int nstats,
									 float8 k1, float8 b,
									 float8 avg_doc_len, int doc_len);
void      bm25_update_idf_stats(const char *chunk_table,
								BM25Term *tokens, int ntokens);
float8    bm25_avg_doc_len_internal(const char *chunk_table);

/* SQL-callable wrappers (PG_FUNCTION_INFO_V1 defined in bm25.c only) */
extern Datum pgedge_vectorizer_bm25_query_vector(PG_FUNCTION_ARGS);
extern Datum pgedge_vectorizer_bm25_avg_doc_len(PG_FUNCTION_ARGS);
extern Datum pgedge_vectorizer_bm25_tokenize(PG_FUNCTION_ARGS);

#endif   /* PGEDGE_BM25_H */
