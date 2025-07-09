package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/koenighotze/rag-demo/internal/query"
	"github.com/tmc/langchaingo/llms/ollama"
)

func createQueryHandler(llm *ollama.LLM, guardRailLlm *ollama.LLM) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			Query string
		}
		err := json.NewDecoder(r.Body).Decode(&request)
		if err != nil {
			log.Printf("Cannot parse request body: %s\n", err.Error())
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		response, err := query.GenerateAnswer(llm, guardRailLlm, request.Query)
		if err != nil {
			w.WriteHeader(http.StatusInternalServerError)
			log.Printf("Cannot generate answer: %s\n", err.Error())
			//nolint:errcheck
			fmt.Fprintf(w, "Sorry, cannot generate an answer at this time! Reason: %s\n", err.Error())
			return
		}

		w.WriteHeader(http.StatusOK)
		log.Printf("Generated response: %s\n", response)
		//nolint:errcheck
		fmt.Fprintf(w, "%s", response)
	}
}

func main() {
	llm, err := ollama.New(ollama.WithModel("deepseek-r1:1.5b"))
	if err != nil {
		log.Default().Fatalln(err)
	}

	guardRailLlm, err := ollama.New(ollama.WithModel("llama-guard3:1b"))
	if err != nil {
		log.Default().Fatalln(err)
	}

	http.HandleFunc("/query", createQueryHandler(llm, guardRailLlm))

	fmt.Println("Starting server on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
