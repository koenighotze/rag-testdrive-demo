package main

import (
	"context"
	"io/fs"
	"log"
	"path/filepath"
	"strings"

	"github.com/koenighotze/rag-demo/internal/embedding"
	"github.com/koenighotze/rag-demo/internal/vectordb"
	"github.com/ledongthuc/pdf"
	"github.com/qdrant/go-client/qdrant"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/textsplitter"
)

func walkTextCorpus(qdrantClient *qdrant.Client) (embeddings.Embedder, error) {
	embedder := embedding.Default()

	return embedder, filepath.WalkDir("text-data-corpus/", func(path string, d fs.DirEntry, err error) error {
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

// TODO DAS HIER SOLLTE VERSCHOBEN WERDEN!
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

	points := vectordb.CreatePointsFromEmbeddings(embeds, path, chunks)

	result, err := qdrantClient.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: "rag",
		Points:         points,
	})

	if result != nil {
		log.Printf("Result of storing chunks: %s", result.Status)
	}

	return err
}

func searchForItem(embedder embeddings.Embedder, client *qdrant.Client, query string) {
	log.Println("SEARCHING FOR ", query)

	embeds, err := embedder.EmbedDocuments(context.Background(), []string{query})
	if err != nil {
		return
	}

	searchResult, err := vectordb.ExecuteSearch(client, embeds[0])
	if err != nil {
		panic(err)
	}
	log.Printf("SEARCH RESULTS")
	if len(searchResult) < 1 {
		log.Println("NOTHING FOUND!!!!")
		return
	}

	log.Printf("Searchresult values: Id %d, Ordervalue %d, Score %f ", searchResult[0].Id.GetNum(), searchResult[0].OrderValue.GetInt(), searchResult[0].Score)
	log.Println(searchResult[0].Payload["chunk"].GetStringValue())
}

func main() {
	client, err := vectordb.Client(true)

	if err != nil {
		log.Panic(err)
	}

	embedder, err := walkTextCorpus(client)
	if err != nil {
		log.Panic(err)
	}

	searchForItem(embedder, client, "Foo")
	searchForItem(embedder, client, " What are the programs goals for moving of the mainframe?")
	searchForItem(embedder, client, "The documentation had to be interpreted by  SMEs, but these individuals were spread too thinly across  multiple teams.")

	defer vectordb.Close()
}
