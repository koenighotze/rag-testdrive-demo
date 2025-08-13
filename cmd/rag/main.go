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

func walkTextCorpus(qdrantClient *qdrant.Client) (embeddings.Embedder, error) {
	llm := newEmbedderModel()

	embedder := newEmbedder(llm)
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

func executeSearch(client *qdrant.Client, search []float32) ([]*qdrant.ScoredPoint, error) {
	searchResult, err := client.Query(context.Background(), &qdrant.QueryPoints{
		CollectionName: "rag",
		Query:          qdrant.NewQuery(search...),
		Filter:         &qdrant.Filter{},
		Params: &qdrant.SearchParams{
			/*
				Exact — turn off approximation and do an exact scan
				If Exact = true, Qdrant will compute exact distances against candidates (implementation details can vary by segment/index).
				Pros: truly exact results.
				Cons: can be much slower on large datasets.
				Typically used for small collections, for validation, or when you absolutely need exactness.
				Rule of thumb: Prefer HNSW with a sensible ef; flip Exact to true only when you must.
			*/
			Exact: qdrant.PtrOf(false),

			/*
				Quantization — control whether to use compressed vectors
				Quantization stores vectors in compressed form to save memory and speed certain stages.
				The QuantizationSearchParams (another struct) usually lets you:
				Ignore/disable quantized data and use full-precision vectors only (slower, most accurate).
				Use quantized data for candidate generation and optionally re-score top hits with full precision to regain accuracy.
				Trade-off: speed & memory vs. precision.
				Common pattern: Use quantized vectors for the coarse search, then rescore a small top-K with full precision for quality.
			*/
			// Quantization: &qdrant.QuantizationSearchParams{},

			/*
				IndexedOnly — search only in indexed (or small) segments
				Qdrant stores data in segments. Some are fully indexed; some may be new (“unindexed”) or still building indexes.
				If IndexedOnly = true, the search skips unindexed/big segments to avoid slow paths.
				Trade-off: you might miss very recent vectors that aren’t indexed yet.
				Use when: you need predictable latency more than absolute completeness (e.g., tight SLAs).
			*/
			IndexedOnly: qdrant.PtrOf(true),

			/*
				HnswEf — beam size for HNSW (approximate) search
				Think of HNSW as exploring a graph of vectors. ef is “how many paths do we keep open while searching.”
				Higher ef ⇒ higher recall (more likely to find the true nearest neighbors) but slower.
				Lower ef ⇒ faster but may miss some true neighbors.
				Use this when you want a good balance without going fully exact.

				When to tweak: If results feel “close but not perfect,” try increasing ef. If latency is too high, lower it.
			*/
			HnswEf: qdrant.PtrOf(uint64(200)),
		},
		ScoreThreshold: qdrant.PtrOf(float32(0.4)),
		WithPayload:    qdrant.NewWithPayloadEnable(true),
	})

	return searchResult, err
}

func searchForItem(embedder embeddings.Embedder, client *qdrant.Client, query string) {
	log.Println("SEARCHING FOR ", query)

	embeds, err := embedder.EmbedDocuments(context.Background(), []string{query})
	if err != nil {
		return
	}

	searchResult, err := executeSearch(client, embeds[0])
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
