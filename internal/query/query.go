package query

import (
	"context"
	"fmt"
	"log"

	"github.com/koenighotze/rag-demo/internal/vectordb"
	"github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

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

func executeSearch(client *qdrant.Client, search []float32) ([]*qdrant.ScoredPoint, error) {
	searchResult, err := client.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "rag",
		Query:          qdrant.NewQuery(search...),
		Params: &qdrant.SearchParams{
			Exact: qdrant.PtrOf(false),
			// IndexedOnly: qdrant.PtrOf(true),
			HnswEf: qdrant.PtrOf(uint64(200)),
		},
		ScoreThreshold: qdrant.PtrOf(float32(0.3)),
		WithPayload:    qdrant.NewWithPayloadEnable(true),
	})

	return searchResult, err
}

func withqdrant(query string) (string, error) {
	embedder := newEmbedder(newEmbedderModel())

	embed, err := embedder.EmbedDocuments(context.Background(), []string{query})
	if err != nil {
		return "", err
	}

	client, err := vectordb.Client(false)
	if err != nil {
		return "", err
	}

	res, err := executeSearch(client, embed[0])

	if err != nil {
		return "", err
	}

	if len(res) < 1 {
		log.Println("No context found for query")
		return "", nil
	}

	return res[0].Payload["chunk"].GetStringValue(), nil
}

func sendToLLM(llm *ollama.LLM, query string) (string, error) {
	log.Printf("Sending query '%s' to LLM\n", query)
	completion, err := llms.GenerateFromSinglePrompt(context.Background(), llm, query, llms.WithTemperature(0))
	if err != nil {
		return "", err
	}
	// log.Printf("LLM answered with '%s'\n", completion)
	return completion, nil
}

func GenerateAnswer(llm *ollama.LLM, guardRailLlm *ollama.LLM, query string) (string, error) {
	log.Printf("Generating answer for query with qdrant: %s", query)

	additionalContext, err := withqdrant(query)

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
	completion, err := sendToLLM(llm, prompt)
	if err != nil {
		return "", err
	}

	sanitizedAnswer, err := ApplyResponseGuardrail(guardRailLlm, completion)

	return sanitizedAnswer, err

}

func GenerateAnswerWithoutQdrant(llm *ollama.LLM, guardRailLlm *ollama.LLM, query string) (string, error) {
	log.Printf("Generating answer for without qdrant query: %s", query)

	completion, err := sendToLLM(llm, query)
	if err != nil {
		return "", err
	}

	return completion, err

	// sanitizedQuery, err := ApplyRequestGuardrail(guardRailLlm, query)
	// if err != nil {
	// 	return "", err
	// }

	// completion, err := sendToLLM(llm, sanitizedQuery)
	// if err != nil {
	// 	return "", err
	// }

	// // sanitizedAnswer, err := ApplyResponseGuardrail(guardRailLlm, completion)

	// return sanitizedAnswer, err
}
