package main

import (
	"io/fs"
	"log"
	"path/filepath"
	"strings"

	"github.com/koenighotze/rag-demo/internal/embedding"
	"github.com/koenighotze/rag-demo/internal/vectordb"
	"github.com/ledongthuc/pdf"
)

func walkTextCorpus(vectorDbClient *vectordb.VectorDbClient) (embedding.Embedder, error) {
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

		return extractTextChunksOnParagraphsFromPdf(vectorDbClient, embedder, path)
	})
}

func extractTextChunksOnParagraphsFromPdf(vectorDbClient *vectordb.VectorDbClient, embedder embedding.Embedder, path string) error {
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
			if err = storeChunks(vectorDbClient, embedder, path, fullText.String()); err != nil {
				log.Printf("Cannot store chunks because of %s", err)
			}
			fullText.Reset()
			continue
		}
	}
	if err = storeChunks(vectorDbClient, embedder, path, fullText.String()); err != nil {
		log.Printf("Cannot store chunks because of %s", err)
	}

	return nil
}

func storeChunks(vectorDbClient *vectordb.VectorDbClient, embedder embedding.Embedder, path string, text string) error {
	items, err := embedder.EmbedAllDocuments(path, text)
	if err != nil {
		return err
	}
	return vectorDbClient.AddPointsToCollection(items)
}

func searchForItem(embedder embedding.Embedder, vectorDbClient *vectordb.VectorDbClient, query string) {
	log.Println("SEARCHING FOR ", query)

	item, err := embedder.EmbedDocument(query)
	if err != nil {
		return
	}

	searchResult, err := vectorDbClient.ExecuteSearch(item.Embedding)
	if err != nil {
		panic(err)
	}
	log.Printf("SEARCH RESULTS")
	if len(searchResult) < 1 {
		log.Println("NOTHING FOUND!!!!")
		return
	}

	log.Printf("Searchresult values: Id %d, Score %f ", searchResult[0].Id, searchResult[0].Score)
	log.Println(searchResult[0].Item)
}

func main() {
	client := vectordb.TruncatingVectorDbClient()

	embedder, err := walkTextCorpus(client)
	if err != nil {
		log.Panic(err)
	}

	searchForItem(embedder, client, "Foo")
	searchForItem(embedder, client, " What are the programs goals for moving of the mainframe?")
	searchForItem(embedder, client, "The documentation had to be interpreted by  SMEs, but these individuals were spread too thinly across  multiple teams.")

	defer client.Close()
}
