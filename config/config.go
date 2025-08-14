package config

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sync"
)

var (
	once   sync.Once
	config Config
	err    error
)

type Config struct {
	Query      Query     `json:"query"`
	ServerAddr string    `json:"server_addr"`
	Embedding  Embedding `json:"embedding"`
}

type Query struct {
	MainModel                string `json:"main_model_name"`
	InputGuardrailModelName  string `json:"input_guardrail_model_name"`
	OutputGuardrailModelName string `json:"output_guardrail_model_name"`
}

type Embedding struct {
	ModelName string `json:"model_name"`
}

func DefaultPath() string {
	// Decide on a sensible default; XDG is common on Linux/macOS.
	// Fallback to working directory for simplicity.
	return "config.json"
}

func Load(path string) (Config, error) {
	b, err := os.ReadFile(filepath.Clean(path))
	if err != nil {
		return Config{}, err
	}
	var cfg Config
	if err := json.Unmarshal(b, &cfg); err != nil {
		return Config{}, err
	}
	if cfg.ServerAddr == "" {
		return Config{}, errors.New("server_addr must be set")
	}
	return cfg, nil
}

func Default() Config {
	once.Do(func() {
		config, err = Load(DefaultPath())
	})

	if err != nil {
		panic(err)
	}

	return config
}
