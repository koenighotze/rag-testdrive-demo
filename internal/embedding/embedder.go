package embedding

import (
	"log"

	"github.com/koenighotze/rag-demo/config"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
)

func newEmbedderModel(embedderModelName string) *ollama.LLM {
	llm, err := ollama.New(ollama.WithModel(embedderModelName))
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

func Default() embeddings.Embedder {
	return NewEmbedder(config.Default().Embedding.ModelName)
}

func NewEmbedder(embedderModelName string) embeddings.Embedder {
	return newEmbedder(newEmbedderModel(embedderModelName))
}
