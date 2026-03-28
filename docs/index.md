# pgEdge Vectorizer

pgEdge Vectorizer is a PostgreSQL extension that automatically chunks text
content and generates vector embeddings using background workers. Vectorizer
provides a seamless integration between your PostgreSQL database and embedding
providers like OpenAI, making it easy to build AI-powered search and retrieval
applications.

pgEdge Vectorizer:

- intelligently splits text into optimal-sized chunks.
- handles embedding generation asynchronously using background workers
  without blocking.
- enables easy switching between OpenAI, Voyage AI, and Ollama.
- processes embeddings efficiently in batches for better API usage.
- automatically retries failed operations with exponential backoff.
- provides built-in views for queue and worker monitoring.
- offers extensive GUC parameters for flexible configuration.

## pgEdge Vectorizer Architecture

pgEdge Vectorizer uses a trigger-based architecture with background workers to
process text asynchronously. The following steps describe the processing flow
from data insertion to embedding storage:

1. A trigger detects INSERT or UPDATE operations on the configured table.
2. The chunking module splits the text into chunks using the configured
   strategy.
3. The system inserts chunk records and queue items into the processing
   queue.
4. Background workers pick up queue items using SKIP LOCKED for concurrent
   processing.
5. The configured provider generates embeddings via its API.
6. The storage layer updates the chunk table with the generated
   embeddings.


## Component Diagram

```
┌─────────────┐
│ Source Table│
└──────┬──────┘
       │ Trigger
       ↓
┌──────────────┐
│   Chunking   │
└──────┬───────┘
       ↓
┌──────────────┐     ┌─────────────┐
│ Chunk Table  │←────┤    Queue    │
└──────────────┘     └──────┬──────┘
       ↑                    │
       │              ┌─────┴──────┐
       │              │  Worker 1  │
       │              │  Worker 2  │
       │              │  Worker N  │
       │              └─────┬──────┘
       │                    │
       │              ┌─────┴──────┐
       └──────────────┤  Provider  │
                      │  (OpenAI)  │
                      └────────────┘
```




## Hybrid Search

Hybrid search combines BM25 sparse keyword ranking with dense vector similarity
and merges the two result lists via **Reciprocal Rank Fusion (RRF)**. It excels
at queries containing specific terms or proper nouns where dense-only retrieval
may miss exact matches.

### Enabling Hybrid Search

Add to `postgresql.conf` and reload (`SELECT pg_reload_conf()`):

```ini
pgedge_vectorizer.enable_hybrid = true

# Optional tuning (defaults shown)
pgedge_vectorizer.bm25_k1 = 1.2    # term-frequency saturation (0.0–3.0)
pgedge_vectorizer.bm25_b  = 0.75   # length normalization (0.0–1.0)
```

Once enabled, background workers automatically populate a `sparse_embedding`
column in each chunk table and maintain a `_idf_stats` table of per-term
document frequencies.

### Usage Example

```sql
-- Enable vectorization (unchanged from normal workflow)
SELECT pgedge_vectorizer.enable_vectorization(
    source_table  := 'articles',
    source_column := 'content'
);

-- Hybrid search with RRF fusion
SELECT source_id, chunk, dense_rank, sparse_rank, rrf_score
FROM pgedge_vectorizer.hybrid_search(
    p_source_table := 'articles'::regclass,
    p_query        := 'keyword ranking BM25',
    p_limit        := 10,
    p_alpha        := 0.7,   -- weight of dense results (0–1)
    p_rrf_k        := 60     -- RRF smoothing constant
);

-- Convenience wrapper
SELECT * FROM pgedge_vectorizer.hybrid_search_simple(
    'articles'::regclass, 'vector similarity search', 5
);
```

### Hybrid Search GUC Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `pgedge_vectorizer.enable_hybrid` | `false` | — | Enable BM25 sparse vectors |
| `pgedge_vectorizer.bm25_k1` | `1.2` | `0.0–3.0` | Term frequency saturation |
| `pgedge_vectorizer.bm25_b` | `0.75` | `0.0–1.0` | Document length normalization |

For more information or to download Vectorizer visit:

- **GitHub**: https://github.com/pgEdge/pgedge-vectorizer
- **Documentation**: (https://docs.pgedge.com/pgedge-vectorizer/)
- **Issues**: https://github.com/pgEdge/pgedge-vectorizer/issues

This software is released under [the PostgreSQL License](LICENCE.md).