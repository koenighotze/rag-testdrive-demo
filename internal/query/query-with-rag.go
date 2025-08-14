package query

import (
	"context"
	"fmt"
	"log"

	"github.com/koenighotze/rag-demo/config"
	"github.com/koenighotze/rag-demo/internal/embedding"
	"github.com/koenighotze/rag-demo/internal/vectordb"
	"github.com/tmc/langchaingo/llms/ollama"
)

// func executeSearch(client *qdrant.Client, search []float32) ([]*qdrant.ScoredPoint, error) {
// 	searchResult, err := client.Query(context.Background(), &qdrant.QueryPoints{
// 		CollectionName: "rag",
// 		Query:          qdrant.NewQuery(search...),
// 		Params: &qdrant.SearchParams{
// 			Exact:  qdrant.PtrOf(false),
// 			HnswEf: qdrant.PtrOf(uint64(200)),
// 		},
// 		ScoreThreshold: qdrant.PtrOf(float32(0.3)),
// 		WithPayload:    qdrant.NewWithPayloadEnable(true),
// 	})

// 	return searchResult, err
// }

func withQdrant(query string) (string, error) {
	embedder := embedding.Default()

	embed, err := embedder.EmbedDocuments(context.Background(), []string{query})
	if err != nil {
		return "", err
	}

	client := vectordb.DefaultClient()

	res, err := vectordb.ExecuteSearch(client, embed[0], vectordb.DefaultQdrantSearchConfig())

	if err != nil {
		return "", err
	}

	if len(res) < 1 {
		log.Println("No context found for query")
		return "", nil
	}

	return res[0].Payload["chunk"].GetStringValue(), nil
}

func GenerateAnswerWithRAG(llm *ollama.LLM, guardRailLlm *ollama.LLM, query string) (string, error) {
	log.Printf("Generating answer for query with qdrant: %s", query)

	additionalContext, err := withQdrant(query)

	if err != nil {
		return "", err
	}

	prompt := fmt.Sprintf(`You are a helpful assistant.
Answer the user using only the context below.

Context:
%s

Question: %s`, additionalContext, query)

	if additionalContext == "" {
		prompt = fmt.Sprintf(`You are a helpful assistant.
Answer the following question:

Question: %s`, query)
	}

	log.Println(query)
	completion, err := sendToLLM(llm, prompt, PromptConfig{temperature: config.Default().Query.MainTemperature})
	if err != nil {
		return "", err
	}

	sanitizedAnswer, err := ApplyResponseGuardrail(guardRailLlm, completion)

	return sanitizedAnswer, err

}
