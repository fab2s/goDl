package nn

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/tensor"
)

// Embedding is a lookup table that maps integer indices to dense vectors.
//
//	emb, _ := nn.NewEmbedding(1000, 64)  // vocab=1000, dim=64
//	// indices is a Variable wrapping an Int64 tensor
//	output := emb.Forward(indices)  // [batch, seqLen] → [batch, seqLen, 64]
type Embedding struct {
	Weight        *Parameter // [numEmbeddings, embeddingDim]
	NumEmbeddings int64
	EmbeddingDim  int64
}

// NewEmbedding creates an Embedding module with random normal initialization.
func NewEmbedding(numEmbeddings, embeddingDim int64, opts ...tensor.Option) (*Embedding, error) {
	wData, err := tensor.RandN([]int64{numEmbeddings, embeddingDim}, opts...)
	if err != nil {
		return nil, err
	}

	return &Embedding{
		Weight:        NewParameter(wData, "weight"),
		NumEmbeddings: numEmbeddings,
		EmbeddingDim:  embeddingDim,
	}, nil
}

// Forward looks up embeddings for the given indices.
// Input: Variable wrapping an Int64 tensor of shape [*] (any shape).
// Output: Variable of shape [*, embeddingDim].
func (e *Embedding) Forward(inputs ...*autograd.Variable) *autograd.Variable {
	indices := inputs[0].Data()
	// Flatten indices for IndexSelect, then reshape output.
	origShape := indices.Shape()
	flat := indices.Reshape([]int64{indices.Numel()})
	if err := flat.Err(); err != nil {
		return autograd.ErrVariable(err)
	}

	out := e.Weight.IndexSelect(0, flat)

	// Reshape to [*origShape, embeddingDim]
	outShape := make([]int64, len(origShape)+1)
	copy(outShape, origShape)
	outShape[len(origShape)] = e.EmbeddingDim
	return out.Reshape(outShape)
}

// Parameters returns the embedding weight.
func (e *Embedding) Parameters() []*Parameter {
	return []*Parameter{e.Weight}
}
