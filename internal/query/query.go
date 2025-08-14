package query

import (
	"log"

	"github.com/tmc/langchaingo/llms/ollama"
)

func GeneratePlainAnswer(llm *ollama.LLM, guardRailLlm *ollama.LLM, query string) (string, error) {
	log.Printf("Generating plain answer: %s", query)

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
