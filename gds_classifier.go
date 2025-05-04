package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"math"
	"time"
	"path/filepath"
	"strings"
	"sync"
	"golang.org/x/sync/errgroup"

	"github.com/sashabaranov/go-openai"
	"gopkg.in/yaml.v3"
)

// Constants
const (
	MaxContentLength = 4000 // Maximum characters to process from each file
	EmbeddingDim    = 1536  // Dimension of OpenAI text-embedding-3-small embeddings

	DefaultEmbeddingProvider = "openai"
	DefaultOpenAIModel      = "text-embedding-3-small" // Default if not overridden
)

// Config holds application configuration
type Config struct {
	SourceDir       string
	OutputDir       string
	MappingFile     string
	Move            bool
	DryRun          bool
	Include         string
	Exclude         string
	Concurrency     int
	OpenAIAPIKey    string
	SupportedExts   []string
	ExcludePatterns []string
	LogFile         string // Path to CSV log file

	// Embedding Provider Configuration
	EmbeddingProvider string // "openai" or "ollama"
	// OpenAI / OpenAI-Compatible Settings
	OpenAIBaseURL     string // Custom base URL for OpenAI or OpenAI-compatible APIs
	OpenAIModel       string // Model name for OpenAI or compatible API
	// Ollama Settings
	OllamaURL         string
	OllamaModel       string
}

// Category represents a category in the GDS system
type Category struct {
	Path        string
	Description string
	IsValid     bool
	Children    map[string]*Category
	Vector      []float32
}

// FileEmbedding represents a file and its embedding
type FileEmbedding struct {
	Path      string
	Embedding []float32
}

// CategoryMatch represents a potential category match for a file
type CategoryMatch struct {
	FilePath     string
	CategoryPath string
	Score        float32
}

// CategoryTree represents the entire GDS category tree
type CategoryTree struct {
	Root       map[string]*Category
	ValidPaths []string
}

// GDSNode represents a node in the GDS YAML
type GDSNode struct {
	Valid    bool                `yaml:"_valid"`
	Children map[string]*GDSNode `yaml:",inline"`
}

func main() {
	// Parse command line arguments
	config := parseArgs()

	// Get API key from environment
	config.OpenAIAPIKey = os.Getenv("OPENAI_API_KEY")

	// --- Log File Setup ---
	var logFile *os.File

	// Check for API key *after* log file setup so we can log the fatal error
	if config.EmbeddingProvider == "openai" && config.OpenAIAPIKey == "" && config.OpenAIBaseURL == "" {
		// Only fatal if using default OpenAI and key is missing.
		// Local models or Ollama might not need it.
		log.Fatal("Error: OPENAI_API_KEY environment variable must be set for OpenAI provider without a custom base URL.")
	}
	var csvWriter *csv.Writer
	if config.LogFile != "" {
		var err error
		logFile, err = os.OpenFile(config.LogFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Fatalf("Error opening log file '%s': %v", config.LogFile, err)
		}
		defer logFile.Close()

		info, err := logFile.Stat()
		if err != nil {
			log.Fatalf("Error getting log file info '%s': %v", config.LogFile, err)
		}

		csvWriter = csv.NewWriter(logFile)

		if info.Size() == 0 {
			header := []string{"path", "filename", "filesize", "category_id", "score"}
			if err := csvWriter.Write(header); err != nil {
				log.Fatalf("Error writing header to log file '%s': %v", config.LogFile, err)
			}
			csvWriter.Flush()
		}
	}

	// Process include extensions
	if config.Include != "" {
		includedExts := strings.Split(config.Include, ",")
		for i, ext := range includedExts {
			ext = strings.TrimSpace(ext)
			if !strings.HasPrefix(ext, ".") {
				includedExts[i] = "." + ext
			} else {
				includedExts[i] = ext
			}
		}
		config.SupportedExts = append(config.SupportedExts, includedExts...)
	}

	// Process exclude patterns
	if config.Exclude != "" {
		config.ExcludePatterns = strings.Split(config.Exclude, ",")
		for i, pattern := range config.ExcludePatterns {
			config.ExcludePatterns[i] = strings.TrimSpace(pattern)
		}
	}

	// Load the GDS category system
	categoryTree, err := loadCategorySystem(config.MappingFile)
	if err != nil {
		log.Fatalf("Error loading category system: %v", err)
	}

	log.Printf("Loaded %d valid category paths", len(categoryTree.ValidPaths))

	log.Printf("Using embedding provider: %s", config.EmbeddingProvider)
	if config.EmbeddingProvider == "openai" {
		log.Printf("OpenAI/Compatible Model: %s", config.OpenAIModel)
		if config.OpenAIBaseURL != "" {
			log.Printf("OpenAI/Compatible Base URL: %s", config.OpenAIBaseURL)
		} else {
			log.Println("Using default OpenAI Base URL.")
		}
	} else if config.EmbeddingProvider == "ollama" {
		log.Printf("Ollama URL: %s, Model: %s", config.OllamaURL, config.OllamaModel)
	}

	// Load or generate embeddings cache
	cacheFile := config.MappingFile + ".embeddings.json"
	yamlInfo, _ := os.Stat(config.MappingFile)
	useCache := false
	if info, err := os.Stat(cacheFile); err == nil {
		if info.ModTime().After(yamlInfo.ModTime()) {
			if err := loadEmbeddingCache(categoryTree, cacheFile); err == nil {
				log.Println("✅ loaded category embeddings from cache")
				useCache = true
			} else {
				log.Printf("⚠️ failed to load cache: %v", err)
			}
		}
	}
	if !useCache {
		// Corrected Log Message Here:
		log.Printf("🔄 generating category embeddings using %s...", config.EmbeddingProvider)
		// Pass the full config, which now includes OpenAIModel and OpenAIBaseURL
		if err := generateCategoryEmbeddings(categoryTree, config); err != nil {
			log.Fatalf("Error generating category embeddings: %v", err)
		}
		if err := saveEmbeddingCache(categoryTree, cacheFile); err != nil {
			log.Printf("⚠️ failed to write embedding cache: %v", err)
		} else {
			log.Printf("✅ wrote embedding cache to %s", cacheFile)
		}
	}

	// Find files to process
	files, err := findFiles(config)
	if err != nil {
		log.Fatalf("Error finding files: %v", err)
	}

	if len(files) == 0 {
		log.Println("No matching files found")
		return
	}

	log.Printf("Found %d files to process", len(files))

	// Process files and get embeddings
	fileEmbeddings, err := processFiles(files, config)
	if err != nil {
		log.Fatalf("Error processing files: %v", err)
	}

	if len(fileEmbeddings) == 0 {
		log.Println("No valid content found in files")
		return
	}

	// Match files to categories
	log.Println("Matching files to categories...")
	matches, err := matchFilesToCategories(fileEmbeddings, categoryTree, config)
	if err != nil {
		log.Fatalf("Error matching files to categories: %v", err)
	}

	// Organize files based on matches
	if err := organizeFiles(matches, config, csvWriter); err != nil {
		log.Fatalf("Error organizing files: %v", err)
	}

	log.Println("Classification complete!")
	if !config.DryRun {
		log.Printf("Files organized in: %s", config.OutputDir)
	}
}

// parseArgs parses command line arguments
func parseArgs() Config {
	sourceDir := flag.String("source", "", "Source directory containing files to classify")
	outputDir := flag.String("output", "classified_files", "Output directory for sorted files")
	mappingFile := flag.String("mapping", "gds.yaml", "Path to GDS category mapping file")
	move := flag.Bool("move", false, "Move files instead of copying")
	dryRun := flag.Bool("dry-run", false, "Show what would be done without actually moving/copying files")
	include := flag.String("include", "", "Additional file extensions to include (comma-separated)")
	exclude := flag.String("exclude", "", "Directories or file patterns to exclude (comma-separated)")
	concurrency := flag.Int("concurrency", 4, "Number of concurrent file processes")
	logFile := flag.String("log-file", "", "Optional: Path to CSV file for logging classification details")

	// Add new flags for embedding provider configuration
	embeddingProvider := flag.String("embedding-provider", DefaultEmbeddingProvider, "Embedding provider ('openai' or 'ollama')")
	openaiBaseURL := flag.String("openai-base-url", "", "Optional: Custom base URL for OpenAI or OpenAI-compatible API endpoint")
	openaiModel := flag.String("openai-model", DefaultOpenAIModel, "Model name for OpenAI or compatible provider")
	ollamaURL := flag.String("ollama-url", "", "URL for Ollama API (if provider is 'ollama')")
	ollamaModel := flag.String("ollama-model", "", "Ollama embedding model name (required if provider is 'ollama')")

	flag.Parse()

	args := flag.Args()
	if len(args) > 0 && *sourceDir == "" {
		*sourceDir = args[0]
	}

	if *sourceDir == "" {
		log.Fatal("Error: Source directory is required")
	}

	// Check if source directory exists
	if _, err := os.Stat(*sourceDir); os.IsNotExist(err) {
		log.Fatalf("Error: Source directory '%s' does not exist", *sourceDir)
	}

	// Check if mapping file exists
	if _, err := os.Stat(*mappingFile); os.IsNotExist(err) {
		log.Fatalf("Error: Mapping file '%s' does not exist", *mappingFile)
	}

	// Validate provider and specific configs
	*embeddingProvider = strings.ToLower(*embeddingProvider)
	if *embeddingProvider != "openai" && *embeddingProvider != "ollama" {
		log.Fatalf("Error: Invalid embedding provider '%s'. Choose 'openai' or 'ollama'.", *embeddingProvider)
	}
	if *embeddingProvider == "ollama" {
		if *ollamaModel == "" {
			log.Fatal("Error: --ollama-model is required when --embedding-provider is 'ollama'")
		}
		if *ollamaURL == "" {
			*ollamaURL = "http://localhost:11434"
			log.Printf("Using default Ollama URL: %s", *ollamaURL)
		}
	}

	// Default supported extensions
	defaultExts := []string{
		".txt", ".md", ".pdf", ".docx", ".doc", ".rtf",
		".csv", ".json", ".py", ".js", ".html", ".css",
		".java", ".c", ".cpp", ".h", ".go", ".rb", ".php",
		".swift", ".kt", ".ts", ".sql",
	}

	// Process include extensions
	if *include != "" {
		includedExts := strings.Split(*include, ",")
		for i, ext := range includedExts {
			ext = strings.TrimSpace(ext)
			if !strings.HasPrefix(ext, ".") {
				includedExts[i] = "." + ext
			} else {
				includedExts[i] = ext
			}
		}
		defaultExts = append(defaultExts, includedExts...)
	}

	// Process exclude patterns
	var excludePatterns []string
	if *exclude != "" {
		excludePatterns = strings.Split(*exclude, ",")
		for i, pattern := range excludePatterns {
			excludePatterns[i] = strings.TrimSpace(pattern)
		}
	}

	return Config{
		SourceDir:     *sourceDir,
		OutputDir:     *outputDir,
		MappingFile:   *mappingFile,
		Move:          *move,
		DryRun:        *dryRun,
		Include:       *include,
		Exclude:       *exclude,
		Concurrency:   *concurrency,
		SupportedExts: defaultExts,
		ExcludePatterns: excludePatterns,
		LogFile:         *logFile,

		// Embedding config
		EmbeddingProvider: *embeddingProvider,
		OpenAIBaseURL:     *openaiBaseURL,
		OpenAIModel:       *openaiModel,
		OllamaURL:         *ollamaURL,
		OllamaModel:       *ollamaModel,
	}
}

// loadCategorySystem loads the GDS category system from YAML
func loadCategorySystem(filePath string) (*CategoryTree, error) {
	// Read the YAML file
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	// Parse the YAML into a map
	var rawMap map[string]interface{}
	if err := yaml.Unmarshal(data, &rawMap); err != nil {
		return nil, err
	}

	// Create the category tree
	tree := &CategoryTree{
		Root:       make(map[string]*Category),
		ValidPaths: []string{},
	}

	// Process the raw map into a structured category tree
	processRawMap(rawMap, "", tree)

	return tree, nil
}

// processRawMap processes a raw map from YAML into a structured category tree
func processRawMap(rawMap map[string]interface{}, parentPath string, tree *CategoryTree) {
	for key, value := range rawMap {
		// Skip metadata keys immediately
		if key == "_valid" || key == "_description" {
			continue
		}

		// Create the path component for this node
		currentKey := key // Use the original key for map lookups later

		// Handle potential comments in keys (optional, but good practice)
		if strings.Contains(key, "#") {
			parts := strings.SplitN(key, "#", 2)
			key = strings.TrimSpace(parts[0])
			if key == "" {
				continue // Skip purely commented out keys
			}
			currentKey = key // Update key if comment was stripped
		}

		// Build the full path for this node
		path := currentKey
		if parentPath != "" {
			// Use filepath.Join for OS-agnostic path construction
			path = filepath.Join(parentPath, currentKey)
		}

		// Check if the value associated with the key is a map (representing a category node)
		if nodeData, ok := value.(map[string]interface{}); ok {

			// --- Extract Metadata ---
			isValid := false
			if v, ok := nodeData["_valid"]; ok {
				if b, ok := v.(bool); ok {
					isValid = b
				} else {
					log.Printf("Warning: Invalid type for '_valid' field in category '%s'. Expected bool.", path)
				}
			}

			description := ""
			if v, ok := nodeData["_description"]; ok {
				if s, ok := v.(string); ok {
					description = s
				} else {
					log.Printf("Warning: Invalid type for '_description' field in category '%s'. Expected string.", path)
				}
			}
			// Provide a default description if none was found in YAML
			if description == "" {
				description = fmt.Sprintf("Category: %s", path) // Default description
			}
			// --- End Extract Metadata ---


			// Create the Category struct
			category := &Category{
				Path:        path,
				Description: description, // Use the extracted or default description
				IsValid:     isValid,
				Children:    make(map[string]*Category),
				Vector:      nil, // Will be populated later
			}

			// Add the category to the tree structure
			if parentPath == "" {
				tree.Root[currentKey] = category
			} else {
				// Find the parent category in the tree
				// Use filepath.Separator for cross-platform compatibility
				parts := strings.Split(parentPath, string(filepath.Separator))
				parent := findCategory(tree.Root, parts)
				if parent != nil {
					parent.Children[currentKey] = category
				} else {
					// This case should ideally not happen if the recursion is correct
					log.Printf("Error: Could not find parent category for path '%s'", parentPath)
				}
			}

			// If this category node is marked as valid, add its path to the list
			if isValid {
				tree.ValidPaths = append(tree.ValidPaths, path)
			}

			// Recursively process the children of this node
			processRawMap(nodeData, path, tree)

		} else if value != nil {
			// Handle cases where a key has a non-map, non-nil value (e.g., scalar)
			// This might indicate a malformed YAML structure according to expectations.
			log.Printf("Warning: Unexpected non-map value for key '%s' at path '%s'. Ignoring.", key, path)
		}
	}
}

// findCategory finds a category in the tree based on path parts
// This is an iterative version, generally preferred over deep recursion.
func findCategory(root map[string]*Category, parts []string) *Category {
	if len(parts) == 0 {
		return nil
	}

	current := root
	var cat *Category
	var ok bool

	for _, part := range parts {
		cat, ok = current[part]
		if !ok {
			return nil // Path segment not found
		}
		current = cat.Children // Move down to the children map for the next level
	}
	return cat // Return the category found at the end of the path
}

// getCategoryByPath gets a category by its full path
func getCategoryByPath(root map[string]*Category, path string) *Category {
	parts := strings.Split(path, string(filepath.Separator))
	return findCategory(root, parts)
}

// generateCategoryEmbeddings generates embeddings for each valid category
func generateCategoryEmbeddings(tree *CategoryTree, config Config) error {
	log.Printf("Generating embeddings for %d valid categories using %s...", len(tree.ValidPaths), config.EmbeddingProvider)

	for _, path := range tree.ValidPaths {
		category := getCategoryByPath(tree.Root, path)
		if category == nil {
			log.Printf("Warning: Could not find category struct for valid path '%s'. Skipping embedding.", path)
			continue
		}

		if len(category.Vector) > 0 {
			continue
		}

		embeddingText := fmt.Sprintf("Category Path: %s\nDescription: %s", category.Path, category.Description)
		if len(embeddingText) > MaxContentLength {
			embeddingText = embeddingText[:MaxContentLength]
		}

		embedding, err := getEmbedding(embeddingText, config)
		if err != nil {
			log.Printf("Error generating embedding for category '%s': %v. Skipping.", path, err)
			continue
		}

		if len(embedding) == 0 {
			log.Printf("Warning: No embedding returned for category '%s'. Skipping.", path)
			continue
		}

		category.Vector = embedding
		log.Printf("Generated embedding for: %s", path)
	}

	log.Println("Category embedding generation complete.")
	return nil
}

// findFiles finds all files with supported extensions in the source directory
func findFiles(config Config) ([]string, error) {
	var files []string

	err := filepath.Walk(config.SourceDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip directories
		if info.IsDir() {
			// Check if directory should be excluded
			for _, pattern := range config.ExcludePatterns {
				if strings.Contains(path, pattern) {
					return filepath.SkipDir
				}
			}
			return nil
		}

		// Check if file should be excluded
		for _, pattern := range config.ExcludePatterns {
			if strings.Contains(path, pattern) {
				return nil
			}
		}

		// Check file extension
		ext := strings.ToLower(filepath.Ext(path))
		for _, supportedExt := range config.SupportedExts {
			if ext == supportedExt {
				// Skip Go entrypoint files
				if filepath.Base(path) == "main.go" {
					return nil
				}
				files = append(files, path)
				break
			}
		}

		return nil
	})

	return files, err
}

// processFiles processes files and returns their embeddings
func processFiles(files []string, config Config) ([]FileEmbedding, error) {
	var (
		fileEmbeddings []FileEmbedding
		mu            sync.Mutex
		g, ctx        = errgroup.WithContext(context.Background())
		sem           = make(chan struct{}, config.Concurrency)
	)

	for _, file := range files {
		file := file // capture loop variable
		g.Go(func() error {
			sem <- struct{}{}
			defer func() { <-sem }()

			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				log.Printf("Processing: %s", file)

				content, err := extractTextFromFile(file)
				if err != nil {
					log.Printf("Error extracting text from %s: %v", file, err)
					return nil // Skip this file but don't fail the whole batch
				}

				if content == "" {
					return nil // Skip empty files
				}

				embedding, err := getEmbedding(content, config)
				if err != nil {
					log.Printf("Error getting embedding for %s: %v", file, err)
					return nil // Skip this file but don't fail the whole batch
				}

				mu.Lock()
				fileEmbeddings = append(fileEmbeddings, FileEmbedding{
					Path:      file,
					Embedding: embedding,
				})
				mu.Unlock()
				return nil
			}
		})
	}

	if err := g.Wait(); err != nil {
		return nil, err
	}
	return fileEmbeddings, nil
}

// extractTextFromFile extracts text content from a file
func extractTextFromFile(filePath string) (string, error) {
	// For simple text-based files, read directly
	ext := strings.ToLower(filepath.Ext(filePath))

	// Handle text files directly
	if isTextFile(ext) {
		content, err := os.ReadFile(filePath)
		if err != nil {
			return "", err
		}

		// Limit content length
		text := string(content)
		if len(text) > MaxContentLength {
			text = text[:MaxContentLength]
		}

		return text, nil
	}

	// For binary files, use external tools (this would require integration with tools like
	// tika-server or using external commands, which is beyond the scope of this example)
	// For simplicity, we'll just read the first part of the file and hope for the best
	content, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	// Try to interpret as text
	text := string(content)
	if len(text) > MaxContentLength {
		text = text[:MaxContentLength]
	}

	return text, nil
}

// isTextFile determines if a file extension is for a text-based file
func isTextFile(ext string) bool {
	textExtensions := map[string]bool{
		".txt": true, ".md": true, ".py": true, ".js": true,
		".html": true, ".css": true, ".java": true, ".c": true,
		".cpp": true, ".h": true, ".go": true, ".rb": true,
		".php": true, ".swift": true, ".kt": true, ".ts": true,
		".sql": true, ".json": true, ".xml": true, ".yaml": true,
		".yml": true, ".sh": true, ".bash": true, ".csv": true,
	}

	return textExtensions[ext]
}

// getEmbedding gets embedding vector for text using the configured provider
func getEmbedding(text string, config Config) ([]float32, error) {
	// Clean and truncate the text
	text = strings.ReplaceAll(text, "\n", " ")
	if len(text) > MaxContentLength {
		text = text[:MaxContentLength]
	}
	if strings.TrimSpace(text) == "" {
		text = "empty content"
	}

	switch config.EmbeddingProvider {
	case "ollama":
		if config.OllamaModel == "" || config.OllamaURL == "" {
			return nil, fmt.Errorf("ollama provider selected but model or URL is missing in config")
		}
		return getOllamaEmbedding(text, config.OllamaURL, config.OllamaModel)
	case "openai":
		return getOpenAIEmbedding(text, config)
	default:
		return nil, fmt.Errorf("unknown embedding provider configured: %s", config.EmbeddingProvider)
	}
}

// matchFilesToCategories matches files to categories based on embedding similarity
func matchFilesToCategories(fileEmbeddings []FileEmbedding, tree *CategoryTree, config Config) ([]CategoryMatch, error) {
	var matches []CategoryMatch

	for _, fe := range fileEmbeddings {
		bestMatch := CategoryMatch{
			FilePath:     fe.Path,
			CategoryPath: "",
			Score:        -1.0,
		}

		// Find best matching category
		for _, path := range tree.ValidPaths {
			category := getCategoryByPath(tree.Root, path)
			if category == nil || category.Vector == nil {
				continue
			}

			// Calculate cosine similarity
			similarity := cosineSimilarity(fe.Embedding, category.Vector)

			if similarity > bestMatch.Score {
				bestMatch.CategoryPath = path
				bestMatch.Score = similarity
			}
		}

		// Add to matches if a good match was found
		if bestMatch.Score > 0.0 {
			matches = append(matches, bestMatch)
			log.Printf("Matched '%s' to category '%s' with score %.4f",
				filepath.Base(bestMatch.FilePath), bestMatch.CategoryPath, bestMatch.Score)
		} else {
			// Use unsorted category for files with no good match
			matches = append(matches, CategoryMatch{
				FilePath:     fe.Path,
				CategoryPath: "00.00/unsorted",
				Score:        0.0,
			})
			log.Printf("No good match for '%s', using unsorted category", filepath.Base(fe.Path))
		}
	}

	return matches, nil
}

// cosineSimilarity calculates the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		// Log a warning if either vector is empty
		log.Print("Warning: Attempting cosine similarity with zero-length vector.")
		return 0.0 // Cosine similarity is undefined, 0.0 is a reasonable default for no similarity
	}
	if len(a) != len(b) {
		// This is the correct place to check for mismatches!
		// Log an error as this indicates a fundamental issue with embedding generation/loading
		log.Printf("Error: Cannot compute cosine similarity for vectors of different lengths (%d vs %d). Ensure all embeddings were generated with the same model.", len(a), len(b))
		return -1.0 // Return -1.0 to indicate an invalid comparison or maximum dissimilarity
	}

	var dotProduct float32
	var normA float32
	var normB float32

	for i := 0; i < len(a); i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = float32(math.Sqrt(float64(normA)))
	normB = float32(math.Sqrt(float64(normB)))

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (normA * normB)
}

// organizeFiles organizes files based on category matches
func organizeFiles(matches []CategoryMatch, config Config, csvLogWriter *csv.Writer) error {
	// Create stats to track
	categoryStats := make(map[string]int)

	for _, match := range matches {
		// --- Logging Logic ---
		if csvLogWriter != nil {
			filePath := match.FilePath
			fileName := filepath.Base(filePath)
			categoryID := match.CategoryPath
			score := match.Score

			// Get file size
			var fileSize int64 = -1
			info, err := os.Stat(filePath)
			if err == nil {
				fileSize = info.Size()
			} else {
				log.Printf("Warning: Could not stat file '%s' for logging size: %v", filePath, err)
			}

			record := []string{
				filePath,
				fileName,
				fmt.Sprintf("%d", fileSize),
				categoryID,
				fmt.Sprintf("%.4f", score),
			}

			if err := csvLogWriter.Write(record); err != nil {
				log.Printf("Warning: Failed to write log record for '%s' to '%s': %v", filePath, config.LogFile, err)
			}
		}
		// Create directory path based on category
		categoryPath := match.CategoryPath
		dirPath := filepath.Join(config.OutputDir, categoryPath)

		// Create the directory if it doesn't exist
		if !config.DryRun {
			if err := os.MkdirAll(dirPath, 0755); err != nil {
				return fmt.Errorf("error creating directory %s: %w", dirPath, err)
			}
		}

		// Get destination file path
		fileName := filepath.Base(match.FilePath)
		destPath := filepath.Join(dirPath, fileName)

		// Copy or move the file
		if !config.DryRun {
			if config.Move {
				if err := moveFile(match.FilePath, destPath); err != nil {
					log.Printf("Error moving %s to %s: %v", match.FilePath, destPath, err)
					continue
				}
				log.Printf("Moved %s to %s", match.FilePath, destPath)
			} else {
				if err := copyFile(match.FilePath, destPath); err != nil {
					log.Printf("Error copying %s to %s: %v", match.FilePath, destPath, err)
					continue
				}
				log.Printf("Copied %s to %s", match.FilePath, destPath)
			}
		} else {
			operation := "Would move"
			if !config.Move {
				operation = "Would copy"
			}
			log.Printf("%s %s to %s", operation, match.FilePath, destPath)
		}

		// Update stats
		categoryStats[categoryPath]++
	}

	// Save stats if not dry run
	if !config.DryRun {
		statsPath := filepath.Join(config.OutputDir, "category_stats.json")
		stats := map[string]interface{}{
			"total_files": len(matches),
			"categories":  categoryStats,
		}

		jsonData, err := json.MarshalIndent(stats, "", "  ")
		if err != nil {
			return fmt.Errorf("error marshaling stats: %w", err)
		}

		if err := os.WriteFile(statsPath, jsonData, 0644); err != nil {
			return fmt.Errorf("error writing stats file: %w", err)
		}

		log.Printf("Wrote stats to %s", statsPath)
	}

	return nil
}

// copyFile copies a file from src to dst preserving permissions
func copyFile(src, dst string) error {
	info, err := os.Stat(src)
	if err != nil {
		return err
	}

	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, info.Mode())
	if err != nil {
		return err
	}
	defer out.Close()

	if _, err = io.Copy(out, in); err != nil {
		return err
	}
	
	// Preserve modification time
	return os.Chtimes(dst, info.ModTime(), info.ModTime())
}

// moveFile moves a file from src to dst
func moveFile(src, dst string) error {
	// Try to rename directly (this works only on the same filesystem)
	err := os.Rename(src, dst)
	if err == nil {
		return nil
	}

	// If rename fails, copy and then remove
	if err := copyFile(src, dst); err != nil {
		return err
	}

	return os.Remove(src)
}

// embedding cache helpers

type embeddingCache map[string][]float32

func saveEmbeddingCache(tree *CategoryTree, cacheFile string) error {
	cache := make(embeddingCache)
	for _, path := range tree.ValidPaths {
		cat := getCategoryByPath(tree.Root, path)
		if cat != nil && len(cat.Vector) > 0 {
			cache[path] = cat.Vector
		}
	}
	data, err := json.MarshalIndent(cache, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(cacheFile, data, 0644)
}

func loadEmbeddingCache(tree *CategoryTree, cacheFile string) error {
	data, err := os.ReadFile(cacheFile)
	if err != nil {
		return err
	}
	cache := make(embeddingCache)
	if err := json.Unmarshal(data, &cache); err != nil {
		return err
	}
	for path, vec := range cache {
		cat := getCategoryByPath(tree.Root, path)
		if cat != nil {
			cat.Vector = vec
		}
	}
	return nil
}
// getOpenAIEmbedding retrieves an embedding vector using the OpenAI library
func getOpenAIEmbedding(text string, config Config) ([]float32, error) {
	if config.OpenAIModel == "" {
		return nil, fmt.Errorf("openai: model name is not configured")
	}

	clientConfig := openai.DefaultConfig(config.OpenAIAPIKey)
	if config.OpenAIBaseURL != "" {
		clientConfig.BaseURL = config.OpenAIBaseURL
	}

	client := openai.NewClientWithConfig(clientConfig)

	resp, err := client.CreateEmbeddings(
		context.Background(),
		openai.EmbeddingRequest{
			Input: []string{text},
			Model: openai.EmbeddingModel(config.OpenAIModel),
		},
	)

	if err != nil {
		return nil, fmt.Errorf("openai: error getting embedding from '%s' for model '%s': %w", 
			clientConfig.BaseURL, config.OpenAIModel, err)
	}

	if len(resp.Data) == 0 || len(resp.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("openai: no embedding data returned from '%s' for model '%s'", 
			clientConfig.BaseURL, config.OpenAIModel)
	}

	return resp.Data[0].Embedding, nil
}

// OllamaEmbeddingRequest represents the request body for Ollama embeddings API
type OllamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

// OllamaEmbeddingResponse represents the response from Ollama embeddings API
type OllamaEmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// getOllamaEmbedding gets embeddings from Ollama API
func getOllamaEmbedding(text, ollamaURL, ollamaModel string) ([]float32, error) {
	// Clean and validate input
	if text = strings.TrimSpace(text); text == "" {
		return nil, fmt.Errorf("empty text provided for embedding")
	}

	// Prepare request body
	reqBody := OllamaEmbeddingRequest{
		Model:  ollamaModel,
		Prompt: text,
	}
	
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	// Create HTTP request with context and timeout
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Construct the full URL properly, handling potential trailing slash
	apiEndpoint := strings.TrimSuffix(ollamaURL, "/") + "/api/embeddings"

	req, err := http.NewRequestWithContext(ctx, "POST", apiEndpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// Make request with retries
	var resp *http.Response
	maxRetries := 3
	retryDelay := 1 * time.Second

	for attempt := 1; attempt <= maxRetries; attempt++ {
		resp, err = http.DefaultClient.Do(req)
		if err == nil && resp.StatusCode == http.StatusOK {
			break // Success
		}

		// Close body on non-success before potential retry
		if resp != nil && resp.Body != nil {
			resp.Body.Close()
		}

		if attempt < maxRetries {
			log.Printf("Ollama request failed (attempt %d/%d): %v. Retrying in %s...", attempt, maxRetries, err, retryDelay*time.Duration(attempt))
			time.Sleep(retryDelay * time.Duration(attempt))
			continue
		}

		// If loop finishes, it means all retries failed.
		// The 'err' variable here holds the error from the *last* attempt.
		if err != nil {
			return nil, fmt.Errorf("ollama API failed after %d attempts: %w", maxRetries, err)
		} else {
			// This case handles non-200 status codes on the last attempt
			return nil, fmt.Errorf("ollama API error after %d attempts (status %d)", maxRetries, resp.StatusCode)
		}
	}
	defer resp.Body.Close()

	// Read and check response
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response body: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama API error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse response
	var result OllamaEmbeddingResponse
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	if len(result.Embedding) == 0 {
		return nil, fmt.Errorf("no embedding returned from Ollama API at %s", ollamaURL)
	}

	return result.Embedding, nil
}
