package embedding

import (
	"context"
	"log"

	"github.com/koenighotze/rag-demo/config"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/textsplitter"
)

type KnowledgeItem struct {
	Embedding      []float32
	SourceDocument string
	Chunk          string
}

func EmbedAllDocuments(embedder embeddings.Embedder, path string, text string) ([]KnowledgeItem, error) {
	if len(text) <= 0 {
		return []KnowledgeItem{}, nil
	}

	chunks, err := textsplitter.NewTokenSplitter().SplitText(text)
	if err != nil {
		return []KnowledgeItem{}, err
	}
	embeds, err := embedder.EmbedDocuments(context.Background(), chunks)
	log.Printf("Generated embeddings with size %d for text of length %d", len(embeds), len(text))
	if err != nil {
		return []KnowledgeItem{}, err
	}

	return embeddingsToKnowledgeItems(embeds, path, chunks), nil
}

func embeddingsToKnowledgeItems(embeds [][]float32, sourceDocument string, chunks []string) []KnowledgeItem {
	var items []KnowledgeItem
	for i, e := range embeds {
		items = append(items, KnowledgeItem{
			Embedding:      e,
			Chunk:          chunks[i],
			SourceDocument: sourceDocument,
		})
	}
	return items
}

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

func NewEmbedder(config config.Embedding) embeddings.Embedder {
	return newEmbedder(newEmbedderModel(config.ModelName))
}

func Default() embeddings.Embedder {
	return NewEmbedder(config.Default().Embedding)
}
