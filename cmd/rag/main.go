package main

import (
	"context"
	"io/fs"
	"log"
	"path/filepath"
	"strings"

	"github.com/google/uuid"
	"github.com/koenighotze/rag-demo/internal/vectordb"
	"github.com/ledongthuc/pdf"
	"github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/textsplitter"
)

func newEmbedderModel() *ollama.LLM {
	llm, err := ollama.New(ollama.WithModel("quentinz/bge-base-zh-v1.5:latest"))
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

func walkTextCorpus(qdrantClient *qdrant.Client) error {
	llm := newEmbedderModel()

	embedder := newEmbedder(llm)
	return filepath.WalkDir("text-data-corpus/", func(path string, d fs.DirEntry, err error) error {
		log.Println("Walking on " + path)

		if !d.Type().IsRegular() {
			log.Printf("Skip %s. Is not a regular file", path)

			return nil
		}

		if filepath.Ext(path) != ".pdf" {
			log.Printf("Skip %s. Is not a PDF file", path)

			return nil
		}

		return extractTextChunksOnParagraphsFromPdf(qdrantClient, embedder, path)
	})
}

func extractTextChunksOnParagraphsFromPdf(qdrantClient *qdrant.Client, embedder embeddings.Embedder, path string) error {
	log.Printf("Processing text in file %s", path)
	file, reader, err := pdf.Open(path)
	if err != nil {
		return err
	}
	defer file.Close() //nolint:errcheck

	// TODO optimize me (max string length and such)
	var fullText strings.Builder
	for pageNumber := range reader.NumPage() {
		log.Printf("Working on page %d", pageNumber)

		page := reader.Page(pageNumber)
		text, err := page.GetPlainText(nil)
		if err != nil {
			log.Printf("Could not get text from page %d because of %s", pageNumber, err)
			continue
		}

		fullText.WriteString(text)

		if fullText.Len() >= 3000 {
			log.Println("Max length of fulltext block reached. Should store chunks now...")
			if err = storeChunks(qdrantClient, embedder, path, fullText.String()); err != nil {
				log.Printf("Cannot store chunks because of %s", err)
			}
			fullText.Reset()
			continue
		}
	}
	if err = storeChunks(qdrantClient, embedder, path, fullText.String()); err != nil {
		log.Printf("Cannot store chunks because of %s", err)
	}

	return nil
}

func createPointsFromEmbeddings(embeds [][]float32, sourceDocument string, chunks []string) []*qdrant.PointStruct {
	var points []*qdrant.PointStruct
	for i, e := range embeds {
		points = append(points, &qdrant.PointStruct{
			Id:      qdrant.NewIDUUID(uuid.New().String()),
			Vectors: qdrant.NewVectors(e...),
			Payload: qdrant.NewValueMap(map[string]any{
				"path":  sourceDocument,
				"chunk": chunks[i],
			}),
		})
	}
	return points
}

func storeChunks(qdrantClient *qdrant.Client, embedder embeddings.Embedder, path string, text string) error {
	if len(text) <= 0 {
		return nil
	}

	chunks, err := textsplitter.NewTokenSplitter().SplitText(text)
	if err != nil {
		return err
	}

	embeds, err := embedder.EmbedDocuments(context.Background(), chunks)
	if err != nil {
		return err
	}

	log.Printf("Generated embeddings with size %d for text of length %d", len(embeds), len(text))

	points := createPointsFromEmbeddings(embeds, path, chunks)

	result, err := qdrantClient.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: "rag",
		Points:         points,
	})

	if result != nil {
		log.Printf("Result of storing chunks: %s", result.Status)
	}

	return err
}

func main() {
	client, err := vectordb.Client()

	if err != nil {
		log.Panic(err)
	}

	if err := walkTextCorpus(client); err != nil {
		log.Panic(err)
	}

	defer vectordb.Close()
}
