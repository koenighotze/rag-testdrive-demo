package vectordb

import (
	"context"
	"log"
	"sync"

	"github.com/qdrant/go-client/qdrant"
)

var (
	once    sync.Once
	client  *qdrant.Client
	initErr error
)

// Client returns the singleton *qdrant.Client.
// The first caller creates the connection and (optionally) the collection.
func Client(truncate bool) (*qdrant.Client, error) {
	once.Do(func() {
		c, err := qdrant.NewClient(&qdrant.Config{
			Host: "localhost",
			Port: 6334,
		})
		if err != nil {
			initErr = err
			return
		}

		if err := ensureCollection(context.Background(), c, "rag", truncate); err != nil {
			initErr = err
			return
		}

		client = c
	})

	return client, initErr
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

func Close() {
	if client == nil {
		return
	}
	if err := client.Close(); err != nil {
		log.Printf("Cannot close qdrant client cleanly. %s", err)
	}
}
