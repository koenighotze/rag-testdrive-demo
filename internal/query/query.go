package query

import (
	"log"

	"github.com/koenighotze/rag-demo/config"
	"github.com/tmc/langchaingo/llms/ollama"
)

func GeneratePlainAnswer(llm *ollama.LLM, guardRailLlm *ollama.LLM, query string) (string, error) {
	log.Printf("Generating plain answer: %s", query)

	sanitizedQuery, err := ApplyRequestGuardrail(guardRailLlm, query)
	if err != nil {
		return "", err
	}

	// TODO refactor
	completion, err := sendToLLM(llm, sanitizedQuery, PromptConfig{temperature: config.Default().Query.MainTemperature})
	if err != nil {
		return "", err
	}

	sanitizedAnswer, err := ApplyResponseGuardrail(guardRailLlm, completion)

	return sanitizedAnswer, err
}
