# goDl development commands
#
# All commands run inside the Docker container via docker compose.
# The project source is mounted at /workspace.

COMPOSE = docker compose
RUN     = $(COMPOSE) run --rm dev

.PHONY: build test test-cpu test-race lint lint-fix cover doc shell clean image env

# Build the Docker image
image:
	$(COMPOSE) build

# Run all tests (CPU + CUDA if available)
test: image
	$(RUN) go test -tags cuda -v ./...

# Run tests without CUDA (skips GPU tests gracefully)
test-cpu: image
	$(RUN) go test -v ./...

# Run tests with verbose output and race detector
test-race: image
	$(RUN) go test -tags cuda -v -race ./...

# Interactive shell in the dev container
shell: image
	$(COMPOSE) run --rm dev bash

# Go vet
vet: image
	$(RUN) go vet -tags cuda ./...

# Lint with golangci-lint
lint: image
	$(RUN) golangci-lint run ./...

# Lint and auto-fix (gofmt + golangci-lint)
lint-fix: image
	$(RUN) sh -c 'gofmt -w . && golangci-lint run --fix ./...'

# Documentation server (pkg.go.dev style) — open http://localhost:6060
doc: image
	@echo "Starting doc server at http://localhost:6060/github.com/fab2s/goDl"
	$(COMPOSE) run --rm -p 127.0.0.1:6060:6060 dev pkgsite -http=0.0.0.0:6060 .

# Test coverage report
cover: image
	$(RUN) sh -c 'go test -coverprofile=cover.out ./... && go tool cover -func=cover.out'

# Clean up containers and volumes
clean:
	$(COMPOSE) down -v --rmi local

# Show Go environment inside the container
env: image
	$(RUN) go env
