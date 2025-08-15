package vectordb

import (
	"context"
	"log"

	"github.com/google/uuid"
	"github.com/koenighotze/rag-demo/config"
	"github.com/koenighotze/rag-demo/internal/embedding"
	"github.com/qdrant/go-client/qdrant"
)

type VectorDbClient struct {
	client *qdrant.Client
}

type QdrantSearchConfig struct {
	Exact          bool
	IndexedOnly    bool
	ScoreThreshold float32
	BeamSize       uint64
}

func defaultQdrantSearchConfig() QdrantSearchConfig {
	return QdrantSearchConfig{
		Exact:          false,
		IndexedOnly:    false,
		BeamSize:       uint64(200),
		ScoreThreshold: float32(0.3),
	}
}

func (c *VectorDbClient) Close() {
	close()
	c.client = nil
}

func (c *VectorDbClient) AddPointsToCollection(items []*embedding.KnowledgeItem) error {
	return c.addPointsToCollection(createPointsFromEmbeddings(items))
}

func (c *VectorDbClient) addPointsToCollection(points []*qdrant.PointStruct) error {
	result, err := c.client.Upsert(context.Background(), &qdrant.UpsertPoints{
		CollectionName: config.Default().Qdrant.CollectionName, // TODO This is bad
		Points:         points,
	})
	if err != nil {
		return err
	}
	if result != nil {
		log.Printf("Result of storing chunks: %s", result.Status)
	}
	return nil
}

func createPointsFromEmbeddings(items []*embedding.KnowledgeItem) []*qdrant.PointStruct {
	var points []*qdrant.PointStruct
	for _, e := range items {
		points = append(points, &qdrant.PointStruct{
			Id:      qdrant.NewIDUUID(uuid.New().String()),
			Vectors: qdrant.NewVectors(e.Embedding...),
			Payload: qdrant.NewValueMap(map[string]any{
				"path":  e.SourceDocument,
				"chunk": e.Chunk,
			}),
		})
	}
	return points
}

type SearchResult struct {
	Id    string
	Score float32
	Item  embedding.KnowledgeItem
}

func (c *VectorDbClient) ExecuteSearch(search []float32) ([]*SearchResult, error) {
	res, err := executeSearch(c.client, search, defaultQdrantSearchConfig())
	if err != nil {
		return nil, err
	}

	var result []*SearchResult
	for _, r := range res {
		result = append(result, &SearchResult{
			Id:    r.Id.GetUuid(),
			Score: r.Score,
			Item: embedding.KnowledgeItem{
				Embedding:      r.Vectors.GetVector().Data,
				Chunk:          r.Payload["chunk"].GetStringValue(),
				SourceDocument: r.Payload["path"].GetStringValue(),
			},
		})
	}
	return result, nil
}

func DefaultVectorDbClient() *VectorDbClient {
	return &VectorDbClient{
		client: defaultClient(),
	}
}

func TruncatingVectorDbClient() *VectorDbClient {
	c, err := newClient(true, config.Default().Qdrant)
	if err != nil {
		log.Panic(err)
	}
	return &VectorDbClient{
		client: c,
	}
}
