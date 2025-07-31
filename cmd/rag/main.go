package main

import (
	"context"
	"log"

	"github.com/google/uuid"
	"github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/llms/ollama"
)

func storeInQdrant(embedder *ollama.LLM) {
	// 	// 1. connect to Qdrant
	client, err := qdrant.NewClient(&qdrant.Config{
		Host: "localhost",
		Port: 6334,
	})

	if err != nil {
		log.Default().Fatalln(err)
	}

	// embedder, err := ollama.New(ollama.WithModel("quentinz/bge-base-zh-v1.5:latest"))
	// if err != nil {
	// 	log.Default().Fatalln(err)
	// }

	// client, err := qdrant.New(
	// 	context.Background(),
	// 	qdrant.WithHost("http://localhost:6333"),
	// 	qdrant.WithCollectionName("test-embeddings"),
	// 	qdrant.WithEmbedder(embedder),
	// )

	// if err != nil {
	// 	log.Default().Fatalln(err)
	// }

	client.CreateCollection(context.Background(), &qdrant.CreateCollection{
		CollectionName: "test",
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     4,
			Distance: qdrant.Distance_Cosine,
		}),
	})

	embedding, err := embedder.CreateEmbedding(context.Background(), []string{"Fee fei fo famm"})
	if err != nil {
		log.Default().Fatalln(err)
	}

	result, err := client.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: "test",
		Points: []*qdrant.PointStruct{
			// {
			// 	Id:      qdrant.NewIDNum(1),
			// 	Vectors: qdrant.NewVectors(0.05, 0.61, 0.76, 0.74),
			// 	Payload: qdrant.NewValueMap(map[string]any{"city": "London"}),
			// },
			{
				Id:      qdrant.NewIDUUID(uuid.New().String()),
				Vectors: qdrant.NewVectors(embedding[0]...),
			},
		},
	})

	if err != nil {
		log.Default().Fatalln(err)
	}

	println(result.Status)

	// // 3. embedder (OpenAI example; swap for self-hosted BGE etc.)
	// embedder, err := embeddings.NewEmbedder(
	// 	embeddings.Providers.OpenAI(os.Getenv("OPENAI_API_KEY")),
	// )
	// if err != nil { log.Fatal(err) }

	// // 4. walk documents
	// err = filepath.Walk("corpus", func(path string, info os.FileInfo, _ error) error {
	// 	// simple splitter 800 chars + overlap
	// 	chunks, _ := textsplitter.SplitRecursive(path, 800, 200)
	// 	for _, ch := range chunks {
	// 		vec, _ := embedder.EmbedQuery(ctx, ch.Text)
	// 		_, err = client.Upsert(ctx, collection, []qc.Point{{ID: ch.ID, Vector: vec,
	// 			Payload: map[string]string{"source": path, "text": ch.Text}}})
	// 	}
	// 	return err
	// })
	// if err != nil { log.Fatal(err) }
	// log.Println("Ingestion complete")
}

func generateSampleEmbedding() *ollama.LLM {
	// this generates a sample embedding using quentinz/bge-base-zh-v1.5:latest

	llm, err := ollama.New(ollama.WithModel("quentinz/bge-base-zh-v1.5:latest"))
	if err != nil {
		log.Default().Fatalln(err)
	}

	embedding, err := llm.CreateEmbedding(context.Background(), []string{"Fee fei fo famm"})
	if err != nil {
		log.Default().Fatalln(err)
	}

	for i, vec := range embedding {
		log.Printf("Embedding %d: %v", i, vec)
	}
	return llm

}

func main() {
	generateSampleEmbedding()

	storeInQdrant(generateSampleEmbedding())
}
