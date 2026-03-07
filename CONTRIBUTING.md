# Contributing to goDl

Thank you for your interest in goDl. Contributions are welcome and appreciated.

## Getting Started

goDl builds against libtorch via CGo, so all development happens inside Docker:

```bash
git clone https://github.com/fab2s/goDl.git
cd goDl
make image      # build dev container (Go + libtorch + CUDA)
make shell      # interactive shell inside the container
make test       # run all tests (CPU + CUDA)
make test-cpu   # run without GPU
make test-race  # run with Go race detector
```

You do **not** need Go or libtorch installed on the host machine.

## Development Workflow

1. Fork the repository and create your branch from `main`.
2. Make your changes inside the dev container (`make shell`).
3. Run `make test` (or `make test-cpu` if you don't have a GPU).
4. Ensure `gofmt` is applied — this is non-negotiable in Go.
5. Open a pull request.

## Code Style

- Standard Go conventions: `gofmt`, `go vet`, no lint warnings.
- Keep the API chainable and consistent with existing patterns.
- Error propagation follows the tensor error chain pattern — read `tensor/tensor.go` to understand it.
- Every differentiable operation needs a backward function and a numerical gradient check in `autograd/gradcheck_test.go`.

## What We're Looking For

**High value contributions:**
- New NN modules (with forward, backward, parameter collection, and gradient checks)
- New autograd operations (with backward and numerical verification)
- Performance improvements to the CGo dispatch path
- Bug fixes with reproducing tests

**Also welcome:**
- Documentation improvements and examples
- Testable examples for pkg.go.dev (`func ExampleLinear()` style)
- CI improvements

**Please discuss first:**
- Changes to public API signatures
- New dependencies
- Architecture changes

Open an issue to discuss before investing significant effort on these.

## Testing

Every PR should pass the existing test suite. If you add new functionality:

- **Tensor ops**: add tests in `tensor/tensor_test.go`
- **Autograd ops**: add a numerical gradient check in `autograd/gradcheck_test.go`
- **NN modules**: add both a functional test in `nn/nn_test.go` and a gradient check in `nn/gradcheck_test.go`
- **Graph features**: add a test in `graph/graph_test.go`

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](./LICENSE).
