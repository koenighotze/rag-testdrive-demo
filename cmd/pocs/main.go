//nolint:all
//go:build ignore
// +build ignore

package main

import (
	"context"
	"log"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
)

func storeInQdrant(embedder embeddings.Embedder) (err error) {
	client, err := qdrant.NewClient(&qdrant.Config{
		Host: "localhost",
		Port: 6334,
	})
	if err != nil {
		return err
	}
	defer client.Close()

	exists, err := client.CollectionExists(context.Background(), "rag")
	if err != nil {
		return err
	}

	if !exists {
		client.CreateCollection(context.Background(), &qdrant.CreateCollection{
			CollectionName: "rag",
			VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
				/*
					This tells Qdrant that each vector (embedding) will have exactly 4 numbers
					Think of it like saying "each data point has 4 dimensions"
					For example: [0.1, -0.5, 0.8, 0.3] - that's 4 numbers
					Important: This must match the actual size of embeddings your model produces
				*/
				Size: 786,
				/*
					This determines how Qdrant calculates similarity between vectors
					Cosine similarity measures the angle between vectors, not their length
					Perfect for text embeddings because it focuses on meaning/direction rather than magnitude
				*/
				Distance: qdrant.Distance_Cosine,
			}),
		})
	}

	embedding, err := embedder.EmbedDocuments(context.Background(), []string{"Fee fei fo famm"})
	if err != nil {
		return err
	}

	result, err := client.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: "rag",
		Points: []*qdrant.PointStruct{
			{
				Id:      qdrant.NewIDUUID(uuid.New().String()),
				Vectors: qdrant.NewVectors(embedding[0]...),
			},
		},
	})

	if err != nil {
		return err
	}

	log.Printf("Upsert result: %s", result.Status)
	return nil
}

func newEmbedderModel() *ollama.LLM {
	llm, err := ollama.New(ollama.WithModel("quentinz/bge-base-zh-v1.5:latest"))
	if err != nil {
		log.Fatalln(err)

	}
	return llm
}

func newEmbedder(embedderClient embeddings.EmbedderClient) embeddings.Embedder {
	embedder, err := embeddings.NewEmbedder(embedderClient, embeddings.WithStripNewLines(true))

	if err != nil {
		log.Fatalln(err)
	}
	return embedder
}

func generateSampleEmbedding() *ollama.LLM {
	llm := newEmbedderModel()

	embedder := newEmbedder(llm)

	embedding, err := embedder.EmbedDocuments(context.Background(), []string{"Fee fei fo famm"})
	if err != nil {
		log.Fatalln(err)
	}

	for i, vec := range embedding {
		log.Printf("Embedding %d: %v", i, vec)
	}

	log.Printf("The model produces embeddings of size: %d", len(embedding[0]))
	return llm
}

func main() {
	generateSampleEmbedding()

	storeInQdrant(newEmbedder(newEmbedderModel()))
}
