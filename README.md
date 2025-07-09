# RAG Tryout

Playground for working with RAG and GenAI using Go-Lang.

## How to run this stuff

### Start the models

`./scripts/run-model.sh`

### Build applications

`make build`

### Start the API service

`./query-service` or run `go run ./cmd/api/query-service.go`

### Run some tests

`./scripts/run-tests.sh`

## TODOs

- logging
- monitoring
- test if models are alive
- refactor

## Draft proposal

Below is a complete walk-through for building an *end-to-end Retrieval-Augmented Generation* (RAG) system in **Go**, with both the LLM **and** the vector database self-hosted on **Google Cloud Run**.

---

### 1.  What is RAG?

Retrieval-Augmented Generation enriches a language-model prompt with documents fetched from a vector store that are semantically closest to the user’s query.
The typical flow is:

1. **Embed** query text → vector
2. **Retrieve** top-*k* similar vectors (and their source texts) from the DB
3. **Augment** a prompt template with those texts
4. **Generate** an answer with the LLM
5. **(optional)** Return citations / sources

---

### 2.  High-level architecture

```text
┌────────────────┐                     ┌─────────────────┐
│  Client / UI   │──HTTP / JSON──────►│  Go RAG API     │
└────────────────┘                     │  (Cloud Run)    │
        ▲                              │  • langchaingo  │
        │                              │  • qdrant-go    │
        │                              └──────┬──────────┘
        │                                     │ gRPC / HTTP
        │                              ┌──────▼──────────┐
        │                              │   Qdrant DB     │
        │                              │ (Cloud Run)     │
        │                              └──────┬──────────┘
        │                                     │ REST
        │                              ┌──────▼──────────┐
        │                              │  LLM (Ollama)   │
        └──────────JSON / stream─────────────►(Cloud Run)│
                                           └─────────────┘
```

---

### 3.  Technology choices

| Layer           | Option (recommended)                                                     | Go package                                                |
| --------------- | ------------------------------------------------------------------------ | --------------------------------------------------------- |
| **Embeddings**  | `langchaingo/embeddings` – supports OpenAI, Gemini, bge-base, Voyage AI… | `github.com/tmc/langchaingo/embeddings` ([pkg.go.dev][1]) |
| **Vector DB**   | **Qdrant** (open-source, GPU-friendly)                                   | `github.com/qdrant/go-client` ([pkg.go.dev][2])           |
| **LLM runtime** | **Ollama** container running Llama 3 / Gemma 2                           | Step-by-step Cloud Run guide ([geshan.com.np][3])         |
| **GPU variant** | Cloud Run GPU tutorial for Gemma 3                                       | ([cloud.google.com][4])                                   |
| **RAG helpers** | `langchaingo` chains, retrievers                                         | GitHub project ([github.com][5])                          |

> **Why Go?**  A single static binary, fast concurrency, easy Cloud-Run containerisation, and the mature `langchaingo` ecosystem.

---

### 4.  Prerequisites

```bash
# Local tooling
go install golang.org/dl/go1.22@latest && go1.22 download
brew install docker gcloud
gcloud init     # authenticate & select project
```

Create two Cloud Run services ahead of time (can be automated with Terraform later):

1. **ollama-llm** – CPU or GPU
2. **qdrant-db** – CPU

The steps in §6 and §7 show the exact commands.

---

### 5.  Go module skeleton

```bash
mkdir ragdemo && cd ragdemo
go mod init github.com/your-org/ragdemo
go get github.com/tmc/langchaingo@v0.3.0 \
       github.com/qdrant/go-client@v1 \
       github.com/go-chi/chi/v5
```

---

### 6.  Deploying the self-hosted LLM (Ollama → Cloud Run)

### 6.1 Dockerfile (`ollama.Dockerfile`)

```dockerfile
FROM ollama/ollama:0.5.7           # latest compatible tag
ENV OLLAMA_MODELS=/models
RUN ollama pull llama3:8b-instruct
EXPOSE 11434
ENTRYPOINT ["ollama", "serve"]
```

### 6.2  Build & deploy

```bash
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/ollama:0.5.7 \
       --file ollama.Dockerfile
gcloud run deploy ollama-llm \
       --image gcr.io/$GOOGLE_CLOUD_PROJECT/ollama:0.5.7 \
       --region europe-west1 \
       --cpu=4 --memory=16Gi \
       --max-instances=2 --concurrency=1 \
       --allow-unauthenticated   # lock down later
```

The blog walk-through confirms volumes, instance sizing and cold-start tips ([geshan.com.np][3]).
For GPU inference, follow Google’s official GPU variant guide and add `--gpu=1` ([cloud.google.com][4]).

---

## 7.  Deploying Qdrant

```bash
cat <<EOF > qdrant.Dockerfile
FROM qdrant/qdrant:v1.9.2
EXPOSE 6333 6334
EOF

gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/qdrant:1.9 \
       --file qdrant.Dockerfile
gcloud run deploy qdrant-db \
       --image gcr.io/$GOOGLE_CLOUD_PROJECT/qdrant:1.9 \
       --region europe-west1 \
       --cpu=2 --memory=4Gi \
       --max-instances=1 --allow-unauthenticated
```

---

## 8.  Ingestion code (`cmd/ingest/main.go`)

```go
package main

import (
	"context"
	"log"
	"os"
	"path/filepath"

	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/textsplitter"
	qc "github.com/qdrant/go-client"
	"github.com/qdrant/go-client/qdrant"
)

const collection = "docs"

func main() {
	ctx := context.Background()

	// 1. connect to Qdrant
	client, err := qc.NewClient(&qc.Config{Host: os.Getenv("QDRANT_HOST"), Port: 6334})
	if err != nil { log.Fatal(err) }

	// 2. ensure collection exists (768-dim cosine)
	_ = client.CreateCollection(ctx, collection, qc.CreateCollectionConfig{
		VectorSize: 768, Distance: qdrant.DistanceCosine,
	})

	// 3. embedder (OpenAI example; swap for self-hosted BGE etc.)
	embedder, err := embeddings.NewEmbedder(
		embeddings.Providers.OpenAI(os.Getenv("OPENAI_API_KEY")),
	)
	if err != nil { log.Fatal(err) }

	// 4. walk documents
	err = filepath.Walk("corpus", func(path string, info os.FileInfo, _ error) error {
		// simple splitter 800 chars + overlap
		chunks, _ := textsplitter.SplitRecursive(path, 800, 200)
		for _, ch := range chunks {
			vec, _ := embedder.EmbedQuery(ctx, ch.Text)
			_, err = client.Upsert(ctx, collection, []qc.Point{{ID: ch.ID, Vector: vec,
				Payload: map[string]string{"source": path, "text": ch.Text}}})
		}
		return err
	})
	if err != nil { log.Fatal(err) }
	log.Println("Ingestion complete")
}
```

*Embedding helpers and API shown in the langchaingo docs* ([pkg.go.dev][1]).
The Qdrant Go client supports `Upsert` and `SearchPoints` via gRPC / REST ([pkg.go.dev][2]).

---

## 9.  Query flow (`internal/rag/service.go`)

```go
type RAGService struct {
	qdrant *qc.Client
	llm    *openai.Client // pointing to Ollama URL with OpenAI-compatible API
	embed  embeddings.Embedder
}

func (s *RAGService) Answer(ctx context.Context, question string) (string, error) {
	qvec, _ := s.embed.EmbedQuery(ctx, question)

	res, _ := s.qdrant.Search(ctx, collection, qvec,
		qc.WithLimit(4), qc.WithScoreThreshold(0.25))

	var contextBuf strings.Builder
	for _, p := range res { contextBuf.WriteString(p.Payload["text"]) }

	prompt := fmt.Sprintf(`You are a helpful assistant.
Answer the user using only the context below.

Context:
%s

Question: %s
Answer:`, contextBuf.String(), question)

	resp, _ := s.llm.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: "llama3:8b-instruct", // matches Ollama tag
		Messages: []openai.ChatCompletionMessage{{Role: "user", Content: prompt}},
	})
	return resp.Choices[0].Message.Content, nil
}
```

---

## 10.  HTTP front-end (`cmd/api/main.go`)

```go
r := chi.NewRouter()
svc := buildService()

r.Post("/chat", func(w http.ResponseWriter, r *http.Request) {
	var req struct{ Query string }
	_ = json.NewDecoder(r.Body).Decode(&req)
	ans, err := svc.Answer(r.Context(), req.Query)
	if err != nil { http.Error(w, err.Error(), 500); return }
	json.NewEncoder(w).Encode(map[string]string{"answer": ans})
})
```

---

## 11.  Containerise & deploy RAG service

```dockerfile
# api.Dockerfile
FROM golang:1.22 as builder
WORKDIR /app
COPY . .
RUN CGO_ENABLED=0 go build -o /ragdemo ./cmd/api

FROM gcr.io/distroless/static
COPY --from=builder /ragdemo /ragdemo
ENV QDRANT_HOST="qdrant-db-<hash>-ew.a.run.app"
ENV LLM_BASE_URL="https://ollama-llm-<hash>-ew.a.run.app"
CMD ["/ragdemo"]
```

```bash
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/rag-api:0.1 \
       --file api.Dockerfile
gcloud run deploy rag-api \
       --image gcr.io/$GOOGLE_CLOUD_PROJECT/rag-api:0.1 \
       --region europe-west1 \
       --cpu=2 --memory=1Gi --min-instances=0 --max-instances=10 \
       --allow-unauthenticated
```

---

## 12.  Basic evaluation

* Use held-out questions and empirical accuracy
* Measure latency (query->answer) and cost (Cloud Run billed CPU seconds)
* Add logging & Prometheus sidecar for QPS, token count, retrieval hits

---

## 13.  Next steps

| Idea                                 | Benefit                           |
| ------------------------------------ | --------------------------------- |
| **Reranking** (e.g., Cohere Re-Rank) | Improves context quality          |
| **Hybrid search** (BM25 + vectors)   | Handles numeric / keyword queries |
| **Streaming responses**              | Lower perceived latency           |
| **Fine-tuned embeddings**            | Domain-specific recall            |
| **Auth on Ollama & Qdrant**          | Zero-trust production deployment  |

---

### Key references

* Step-by-step Ollama on Cloud Run ([geshan.com.np][3])
* Official Cloud Run GPU guide ([cloud.google.com][4])
* RAG in Go with Qdrant article ([yuniko.software][6])
* `langchaingo/embeddings` package ([pkg.go.dev][1])
* Qdrant Go client docs ([pkg.go.dev][2])
* Intro to RAG with Go & langchaingo (AWS Community) ([community.aws][7])

With this blueprint you can stand up a production-ready, fully self-hosted RAG stack in **Go** while keeping all models and data inside your own Google Cloud project.

[1]: https://pkg.go.dev/github.com/tmc/langchaingo/embeddings "embeddings package - github.com/tmc/langchaingo/embeddings - Go Packages"
[2]: https://pkg.go.dev/github.com/qdrant/go-client?utm_source=chatgpt.com "go-client module - github.com/qdrant/go-client - Go Packages"
[3]: https://geshan.com.np/blog/2025/01/ollama-google-cloud-run/ "How to run (any) open LLM with Ollama on Google Cloud Run [Step-by-step]"
[4]: https://cloud.google.com/run/docs/tutorials/gpu-gemma-with-ollama "Run LLM inference on Cloud Run GPUs with Gemma 3 and Ollama  |  Cloud Run Documentation  |  Google Cloud"
[5]: https://github.com/tmc/langchaingo?utm_source=chatgpt.com "GitHub - tmc/langchaingo: LangChain for Go, the easiest way to write ..."
[6]: https://yuniko.software/rag-in-go/?utm_source=chatgpt.com "RAG in Go: A Practical Implementation Using Qdrant and OpenAI"
[7]: https://community.aws/content/2f1mRXuakNO22izRKDVNRazzxhb/how-to-use-retrieval-augmented-generation-rag-for-go-applications?utm_source=chatgpt.com "How to use Retrieval Augmented Generation (RAG) for Go applications"
