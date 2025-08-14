package query

import (
	"context"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

type PromptConfig struct {
	temperature float64
}

func sendToLLM(llm *ollama.LLM, query string, config PromptConfig) (string, error) {
	log.Printf("Sending query '%s' to LLM\n", query)
	completion, err := llms.GenerateFromSinglePrompt(context.Background(), llm, query, llms.WithTemperature(config.temperature))
	if err != nil {
		return "", err
	}
	log.Printf("LLM answered with '%s'\n", completion)
	return completion, nil
}
