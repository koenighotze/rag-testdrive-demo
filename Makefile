.DEFAULT_GOAL := build

.PHONY: all build test vet clean

install.tools:
	go install golang.org/x/vuln/cmd/govulncheck@latest

clean:
	go clean -x -i

fmt:
	go fmt ./cmd/... ./internal/... ./pkg/...

vet: fmt
	go vet ./cmd/... ./internal/... ./pkg/...

lint:
	golangci-lint run ./...

deps.upgrade:
	go get -u ./...
	go mod tidy

deps.vendor:
	go mod vendor

deps.vulncheck:
	govulncheck ./...

deps.nancy:
	go list -json -deps ./... | docker run --rm -i sonatypecommunity/nancy:latest sleuth

get.dependencies:
	go mod tidy

test: get.dependencies
	go test ./internal/... ./cmd/... ./pkg/... -coverprofile=coverage.out

test.all: get.dependencies
	go test -tags=integration ./internal/... ./cmd/... ./pkg/... -coverprofile=coverage.out

test.report: test
	go tool cover -html=coverage.out

build: get.dependencies
	go build -o rag-demo ./cmd/rag-demo

run.local:
	go run ./cmd/rag-demo/main.go
