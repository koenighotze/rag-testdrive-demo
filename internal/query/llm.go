package query

import (
	"context"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

func sendToLLM(llm *ollama.LLM, query string) (string, error) {
	log.Printf("Sending query '%s' to LLM\n", query)
	completion, err := llms.GenerateFromSinglePrompt(context.Background(), llm, query, llms.WithTemperature(0))
	if err != nil {
		return "", err
	}
	log.Printf("LLM answered with '%s'\n", completion)
	return completion, nil
}
