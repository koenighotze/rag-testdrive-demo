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

type Embedder struct {
	embedder embeddings.Embedder
}

func (e *Embedder) EmbedDocument(text string) (*KnowledgeItem, error) {
	embedding, err := e.embedder.EmbedDocuments(context.Background(), []string{text})

	if err != nil {
		return nil, err
	}

	return embeddingToKowledgeItem(embedding[0], "", text), nil
}

func (e *Embedder) EmbedAllDocuments(path string, text string) ([]*KnowledgeItem, error) {
	if len(text) <= 0 {
		return []*KnowledgeItem{}, nil
	}

	chunks, err := textsplitter.NewTokenSplitter().SplitText(text)
	if err != nil {
		return nil, err
	}
	embeds, err := e.embedder.EmbedDocuments(context.Background(), chunks)
	log.Printf("Generated embeddings with size %d for text of length %d", len(embeds), len(text))
	if err != nil {
		return nil, err
	}

	return embeddingsToKnowledgeItems(embeds, path, chunks), nil
}

func embeddingToKowledgeItem(embedding []float32, sourceDocument string, chunk string) *KnowledgeItem {
	return &KnowledgeItem{
		Embedding:      embedding,
		Chunk:          chunk,
		SourceDocument: sourceDocument,
	}
}

func embeddingsToKnowledgeItems(embeds [][]float32, sourceDocument string, chunks []string) []*KnowledgeItem {
	var items []*KnowledgeItem
	for i, e := range embeds {
		items = append(items, embeddingToKowledgeItem(e, sourceDocument, chunks[i]))
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

func NewEmbedder(config config.Embedding) Embedder {
	return Embedder{
		embedder: newEmbedder(newEmbedderModel(config.ModelName)),
	}
}

func Default() Embedder {
	return NewEmbedder(config.Default().Embedding)
}
