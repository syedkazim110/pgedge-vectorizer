-- hybrid_test.sql
-- Regression tests for hybrid BM25 + dense vector search feature.

---------------------------------------------------------------------------
-- Setup
---------------------------------------------------------------------------

CREATE TABLE hybrid_test_docs (
    id      BIGSERIAL PRIMARY KEY,
    content TEXT
);

-- Use a fixed embedding dimension to avoid needing an API key
SELECT pgedge_vectorizer.enable_vectorization(
    'hybrid_test_docs'::regclass,
    'content',
    'token_based',
    100,
    10,
    1536
);

---------------------------------------------------------------------------
-- Test 1: GUC parameters exist
---------------------------------------------------------------------------

-- Verify all three hybrid GUC params are registered
SELECT name
FROM pg_settings
WHERE name IN (
    'pgedge_vectorizer.enable_hybrid',
    'pgedge_vectorizer.bm25_k1',
    'pgedge_vectorizer.bm25_b'
)
ORDER BY name;

---------------------------------------------------------------------------
-- Test 2: sparse_embedding column exists in chunk table
---------------------------------------------------------------------------

SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name   = 'hybrid_test_docs_content_chunks'
  AND column_name  = 'sparse_embedding';

---------------------------------------------------------------------------
-- Test 3: bm25_doc_len generated column exists in chunk table
---------------------------------------------------------------------------

SELECT column_name, is_generated
FROM information_schema.columns
WHERE table_name  = 'hybrid_test_docs_content_chunks'
  AND column_name = 'bm25_doc_len';

---------------------------------------------------------------------------
-- Test 4: IDF stats table was created
---------------------------------------------------------------------------

SELECT tablename
FROM pg_tables
WHERE tablename = 'hybrid_test_docs_content_chunks_idf_stats';

---------------------------------------------------------------------------
-- Test 5: HNSW sparse index was created
---------------------------------------------------------------------------

SELECT indexname
FROM pg_indexes
WHERE tablename = 'hybrid_test_docs_content_chunks'
  AND indexname LIKE '%sparse%';

---------------------------------------------------------------------------
-- Test 6: vectorizers registry was populated
---------------------------------------------------------------------------

SELECT source_table, source_column, chunk_table
FROM pgedge_vectorizer.vectorizers
WHERE source_table = 'hybrid_test_docs';

---------------------------------------------------------------------------
-- Test 7: hybrid_search function exists
---------------------------------------------------------------------------

SELECT proname
FROM pg_proc
WHERE proname = 'hybrid_search'
  AND pronamespace = (
      SELECT oid FROM pg_namespace
      WHERE nspname = 'pgedge_vectorizer'
  );

---------------------------------------------------------------------------
-- Test 8: hybrid_search_simple function exists
---------------------------------------------------------------------------

SELECT proname
FROM pg_proc
WHERE proname = 'hybrid_search_simple'
  AND pronamespace = (
      SELECT oid FROM pg_namespace
      WHERE nspname = 'pgedge_vectorizer'
  );

---------------------------------------------------------------------------
-- Test 9: BM25 tokenizer returns non-empty output for normal text
---------------------------------------------------------------------------

SELECT array_length(
    pgedge_vectorizer.bm25_tokenize('the quick brown fox jumps'),
    1
) > 0 AS has_tokens;

---------------------------------------------------------------------------
-- Test 10: BM25 tokenizer strips stopwords
-- 'the', 'is', 'a' are stopwords; 'cat' and 'test' should remain
---------------------------------------------------------------------------

SELECT token
FROM unnest(pgedge_vectorizer.bm25_tokenize('the cat is a test')) AS token
ORDER BY token;

---------------------------------------------------------------------------
-- Test 11: BM25 tokenizer handles empty string
---------------------------------------------------------------------------

SELECT COALESCE(
    array_length(pgedge_vectorizer.bm25_tokenize(''), 1),
    0
) AS token_count;

---------------------------------------------------------------------------
-- Test 12: hybrid_search raises a clear exception for unknown table
---------------------------------------------------------------------------

DO $$
BEGIN
    PERFORM pgedge_vectorizer.hybrid_search(
        'pg_class'::regclass, 'test query', 5
    );
    RAISE EXCEPTION 'Expected exception was not raised';
EXCEPTION
    WHEN OTHERS THEN
        IF sqlerrm LIKE '%No vectorizer found%' THEN
            RAISE NOTICE 'Got expected exception: %', sqlerrm;
        ELSE
            RAISE;
        END IF;
END;
$$;

---------------------------------------------------------------------------
-- Cleanup
---------------------------------------------------------------------------

SELECT pgedge_vectorizer.disable_vectorization(
    'hybrid_test_docs'::regclass, 'content', true
);

DROP TABLE hybrid_test_docs;
