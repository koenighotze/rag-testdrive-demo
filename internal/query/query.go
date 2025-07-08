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

func GenerateAnswer(llm *ollama.LLM, guardRailLlm *ollama.LLM, query string) (string, error) {
	log.Printf("Generating answer for query: %s", query)

	sanitizedQuery, err := ApplyRequestGuardrail(guardRailLlm, query)
	if err != nil {
		return "", err
	}

	completion, err := sendToLLM(llm, sanitizedQuery)
	if err != nil {
		return "", err
	}

	sanitizedAnswer, err := ApplyResponseGuardrail(guardRailLlm, completion)

	return sanitizedAnswer, err
}
