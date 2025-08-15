package query

import (
	"fmt"
	"log"

	"github.com/koenighotze/rag-demo/config"
	"github.com/koenighotze/rag-demo/internal/embedding"
	"github.com/koenighotze/rag-demo/internal/vectordb"
	"github.com/tmc/langchaingo/llms/ollama"
)

func withQdrant(query string) (string, error) {
	embedder := embedding.Default()

	item, err := embedder.EmbedDocument(query)
	if err != nil {
		return "", err
	}

	client := vectordb.DefaultVectorDbClient()

	res, err := client.ExecuteSearch(item.Embedding)

	if err != nil {
		return "", err
	}

	if len(res) < 1 {
		log.Println("No context found for query")
		return "", nil
	}

	return res[0].Item.Chunk, nil
}

func GenerateAnswerWithRAG(llm *ollama.LLM, guardRailLlm *ollama.LLM, query string) (string, error) {
	log.Printf("Generating answer for query with qdrant: %s", query)

	additionalContext, err := withQdrant(query)

	if err != nil {
		return "", err
	}

	prompt := fmt.Sprintf(`You are a helpful assistant.
Answer the user and consider the context below as your primary context.

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
