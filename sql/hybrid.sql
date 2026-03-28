-- hybrid.sql
-- Hybrid BM25 + dense vector search using Reciprocal Rank Fusion (RRF).
--
-- Requires: pgedge_vectorizer.enable_hybrid = true in postgresql.conf
-- and pgvector >= 0.7.0 for sparsevec support.

---------------------------------------------------------------------------
-- hybrid_search() — main user-facing function
---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION pgedge_vectorizer.hybrid_search(
    p_source_table   REGCLASS,
    p_query          TEXT,
    p_limit          INT     DEFAULT 10,
    p_alpha          FLOAT8  DEFAULT 0.7,
    p_rrf_k          INT     DEFAULT 60
)
RETURNS TABLE (
    source_id   BIGINT,
    chunk       TEXT,
    dense_rank  INT,
    sparse_rank INT,
    rrf_score   FLOAT8
)
LANGUAGE plpgsql AS $$
DECLARE
    v_chunk_table  TEXT;
    v_query_dense  vector;
    v_query_sparse sparsevec;
BEGIN
    -- Look up the chunk table name from the vectorizers registry
    SELECT vz.chunk_table INTO v_chunk_table
    FROM pgedge_vectorizer.vectorizers vz
    WHERE vz.source_table = p_source_table::TEXT
    LIMIT 1;

    IF v_chunk_table IS NULL THEN
        RAISE EXCEPTION
            'No vectorizer found for table %. '
            'Call pgedge_vectorizer.enable_vectorization() first.',
            p_source_table;
    END IF;

    -- Generate dense query vector via the existing C function
    v_query_dense := pgedge_vectorizer.generate_embedding(p_query);

    -- Generate sparse BM25 query vector
    v_query_sparse := pgedge_vectorizer.bm25_query_vector(
                          p_query, v_chunk_table);

    -- Run both ranked lists and merge with Reciprocal Rank Fusion
    RETURN QUERY EXECUTE format($sql$
        WITH dense AS (
            SELECT
                source_id,
                content AS chunk,
                ROW_NUMBER() OVER (
                    ORDER BY embedding <=> %L::vector
                ) AS rnk
            FROM %I
            WHERE embedding IS NOT NULL
            LIMIT %s * 3
        ),
        sparse AS (
            SELECT
                source_id,
                content AS chunk,
                ROW_NUMBER() OVER (
                    ORDER BY sparse_embedding <#> %L::sparsevec ASC
                ) AS rnk
            FROM %I
            WHERE sparse_embedding IS NOT NULL
            LIMIT %s * 3
        ),
        merged AS (
            SELECT
                COALESCE(d.source_id, s.source_id)  AS source_id,
                COALESCE(d.chunk,     s.chunk)       AS chunk,
                COALESCE(d.rnk, 9999)::INT           AS dense_rank,
                COALESCE(s.rnk, 9999)::INT           AS sparse_rank,
                (
                      %s::float8  / (%s + COALESCE(d.rnk, 9999))
                    + (1.0 - %s::float8) / (%s + COALESCE(s.rnk, 9999))
                )                                    AS rrf_score
            FROM dense d
            FULL OUTER JOIN sparse s USING (source_id)
        )
        SELECT
            source_id::bigint,
            chunk,
            dense_rank,
            sparse_rank,
            rrf_score
        FROM merged
        ORDER BY rrf_score DESC
        LIMIT %s
    $sql$,
        v_query_dense,   v_chunk_table, p_limit,
        v_query_sparse,  v_chunk_table, p_limit,
        p_alpha, p_rrf_k,
        p_alpha, p_rrf_k,
        p_limit
    );
END;
$$;

COMMENT ON FUNCTION pgedge_vectorizer.hybrid_search IS
'Hybrid BM25 + dense vector search using Reciprocal Rank Fusion.
 p_alpha controls the weight of dense results (0 = pure sparse, 1 = pure dense).
 p_rrf_k is the RRF rank smoothing constant (default 60).
 Requires pgedge_vectorizer.enable_hybrid = true in postgresql.conf.';

---------------------------------------------------------------------------
-- hybrid_search_simple() — convenience wrapper
---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION pgedge_vectorizer.hybrid_search_simple(
    p_source_table REGCLASS,
    p_query        TEXT,
    p_limit        INT DEFAULT 10
)
RETURNS TABLE (
    source_id  BIGINT,
    chunk      TEXT,
    rrf_score  FLOAT8
)
LANGUAGE sql AS $$
    SELECT source_id, chunk, rrf_score
    FROM pgedge_vectorizer.hybrid_search(
             p_source_table, p_query, p_limit);
$$;

COMMENT ON FUNCTION pgedge_vectorizer.hybrid_search_simple IS
'Convenience wrapper for hybrid_search() returning only source_id, chunk, and rrf_score';
