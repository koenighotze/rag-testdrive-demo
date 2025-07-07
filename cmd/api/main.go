package main

import (
	"context"
	"fmt"
	"log"
	"net/http"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

func createQueryHandler(llm *ollama.LLM) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		completion, err := llms.GenerateFromSinglePrompt(context.Background(), llm, "hello and good morning!", llms.WithTemperature(0.8))

		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintf(w, "Boom: %s", err.Error())
			return
		}

		fmt.Fprintf(w, "%s", completion)
	}
}

func main() {
	llm, err := ollama.New(ollama.WithModel("deepseek-r1:1.5b"))
	if err != nil {
		log.Default().Fatalln(err)
	}

	http.HandleFunc("/query", createQueryHandler(llm))

	fmt.Println("Starting server on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
