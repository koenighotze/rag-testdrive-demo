package vectordb

import (
	"context"
	"log"
	"sync"

	"github.com/koenighotze/rag-demo/config"
	"github.com/qdrant/go-client/qdrant"
)

var (
	once    sync.Once
	client  *qdrant.Client
	initErr error
)

func executeSearch(client *qdrant.Client, search []float32, searchConfig QdrantSearchConfig) ([]*qdrant.ScoredPoint, error) {
	searchResult, err := client.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "rag",
		Query:          qdrant.NewQuery(search...),
		Filter:         &qdrant.Filter{},
		Params: &qdrant.SearchParams{
			/*
				Exact — turn off approximation and do an exact scan
				If Exact = true, Qdrant will compute exact distances against candidates (implementation details can vary by segment/index).
				Pros: truly exact results.
				Cons: can be much slower on large datasets.
				Typically used for small collections, for validation, or when you absolutely need exactness.
				Rule of thumb: Prefer HNSW with a sensible ef; flip Exact to true only when you must.
			*/
			Exact: qdrant.PtrOf(searchConfig.Exact),

			/*
				Quantization — control whether to use compressed vectors
				Quantization stores vectors in compressed form to save memory and speed certain stages.
				The QuantizationSearchParams (another struct) usually lets you:
				Ignore/disable quantized data and use full-precision vectors only (slower, most accurate).
				Use quantized data for candidate generation and optionally re-score top hits with full precision to regain accuracy.
				Trade-off: speed & memory vs. precision.
				Common pattern: Use quantized vectors for the coarse search, then rescore a small top-K with full precision for quality.
			*/
			// Quantization: &qdrant.QuantizationSearchParams{},

			/*
				IndexedOnly — search only in indexed (or small) segments
				Qdrant stores data in segments. Some are fully indexed; some may be new (“unindexed”) or still building indexes.
				If IndexedOnly = true, the search skips unindexed/big segments to avoid slow paths.
				Trade-off: you might miss very recent vectors that aren’t indexed yet.
				Use when: you need predictable latency more than absolute completeness (e.g., tight SLAs).
			*/
			IndexedOnly: qdrant.PtrOf(searchConfig.IndexedOnly),

			/*
				HnswEf — beam size for HNSW (approximate) search
				Think of HNSW as exploring a graph of vectors. ef is “how many paths do we keep open while searching.”
				Higher ef ⇒ higher recall (more likely to find the true nearest neighbors) but slower.
				Lower ef ⇒ faster but may miss some true neighbors.
				Use this when you want a good balance without going fully exact.

				When to tweak: If results feel “close but not perfect,” try increasing ef. If latency is too high, lower it.
			*/
			HnswEf: qdrant.PtrOf(searchConfig.BeamSize),
		},
		ScoreThreshold: qdrant.PtrOf(searchConfig.ScoreThreshold),
		WithPayload:    qdrant.NewWithPayloadEnable(true),
	})

	return searchResult, err
}

func ensureCollection(ctx context.Context, c *qdrant.Client, name string, truncate bool) error {
	exists, err := c.CollectionExists(ctx, name)
	if err != nil {
		return err
	}
	if exists {
		if !truncate {
			return nil
		}

		log.Println("Truncating collection", name)
		//nolint:errcheck
		c.DeleteCollection(context.Background(), name)
	}

	log.Println("Creating collection: ", name)
	return c.CreateCollection(context.Background(), &qdrant.CreateCollection{
		CollectionName: name,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			/*
				This tells Qdrant that each vector (embedding) will have exactly 4 numbers
				Think of it like saying "each data point has 4 dimensions"
				For example: [0.1, -0.5, 0.8, 0.3] - that's 4 numbers
				Important: This must match the actual size of embeddings your model produces
			*/
			Size: 768,
			/*
				This determines how Qdrant calculates similarity between vectors
				Cosine similarity measures the angle between vectors, not their length
				Perfect for text embeddings because it focuses on meaning/direction rather than magnitude
			*/
			Distance: qdrant.Distance_Cosine,
		}),
	})
}

func defaultClient() *qdrant.Client {
	client, err := newClient(false, config.QdrantConfig())

	if err != nil {
		log.Panic(err)
	}

	return client
}

func newClient(truncate bool, config config.Qdrant) (*qdrant.Client, error) {
	once.Do(func() {
		client, err := qdrant.NewClient(&qdrant.Config{
			Host: config.Host,
			Port: config.Port,
		})

		if err != nil {
			initErr = err
			return
		}

		if err := ensureCollection(context.Background(), client, config.CollectionName, truncate); err != nil {
			initErr = err
			return
		}
	})

	return client, initErr
}

func Close() {
	if client == nil {
		return
	}
	if err := client.Close(); err != nil {
		log.Printf("Cannot close qdrant client cleanly. %s", err)
	}
}
