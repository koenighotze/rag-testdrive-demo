package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/koenighotze/rag-demo/config"
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

		response, err := query.GeneratePlainAnswer(llm, guardRailLlm, request.Query)
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

func createRagQueryHandler(llm *ollama.LLM, guardRailLlm *ollama.LLM) http.HandlerFunc {
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

		response, err := query.GenerateAnswerWithRAG(llm, guardRailLlm, request.Query)
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
	config := config.Default()
	log.Println(config)

	llm, err := ollama.New(ollama.WithModel(config.Query.MainModel))
	if err != nil {
		log.Default().Fatalln(err)
	}

	guardRailLlm, err := ollama.New(ollama.WithModel(config.Query.InputGuardrailModelName))
	if err != nil {
		log.Default().Fatalln(err)
	}

	http.HandleFunc("/query", createQueryHandler(llm, guardRailLlm))
	http.HandleFunc("/ragquery", createRagQueryHandler(llm, guardRailLlm))

	fmt.Println("Starting server on ", config.ServerAddr)
	log.Fatal(http.ListenAndServe(config.ServerAddr, nil))
}
