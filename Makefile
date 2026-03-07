# goDl development commands
#
# All commands run inside the Docker container via docker compose.
# The project source is mounted at /workspace.

COMPOSE = docker compose
RUN     = $(COMPOSE) run --rm dev

.PHONY: build test test-cpu test-race lint lint-fix shell clean image env

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

# Clean up containers and volumes
clean:
	$(COMPOSE) down -v --rmi local

# Show Go environment inside the container
env: image
	$(RUN) go env
