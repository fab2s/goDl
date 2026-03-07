// Package showcase demonstrates every method of the graph fluent builder
// API in a single coherent graph.
//
// Builder methods exercised:
//
//	From, Through, Tag, Using (backward ref), Using (forward ref),
//	Split, Merge, Also, Map.Slices, Map.Each, Map.Over,
//	Loop.For, Loop.While, Loop.Until,
//	Gate, Gate.Using, Switch, Switch.Using, Build
//
// Graph methods exercised:
//
//	Forward, Parameters, SetTraining, ResetState, DetachState
//
// Also demonstrates Graph-as-Module: sub-graphs used as reusable blocks
// inside Split branches and Loop bodies.
package showcase

import (
	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/graph"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// ffnBlock builds a reusable feed-forward sub-graph: Linear → GELU → LayerNorm.
// Each call creates fresh parameters — sub-graphs compose like any module.
func ffnBlock(dim int64) *graph.Graph {
	g, err := graph.From(nn.MustLinear(dim, dim)).
		Through(nn.NewGELU()).
		Through(nn.MustLayerNorm(dim)).
		Build()
	if err != nil {
		panic("ffnBlock: " + err.Error())
	}
	return g
}

// readHead builds a projection head sub-graph: Linear → LayerNorm.
// Used by Split to create independent read heads over the same input.
func readHead(dim int64) *graph.Graph {
	g, err := graph.From(nn.MustLinear(dim, dim)).
		Through(nn.MustLayerNorm(dim)).
		Build()
	if err != nil {
		panic("readHead: " + err.Error())
	}
	return g
}

// heavyPathSelector is a user-defined Switch selector.
// It examines the "refined" state (received via Using) and picks:
//   - branch 0 (lightweight Linear) when activations are small
//   - branch 1 (full block sub-graph) when activations are large
//
// Implements nn.NamedInputModule for named access to Using refs,
// and nn.RefValidator for build-time validation of expected refs.
type heavyPathSelector struct{}

// Forward is the nn.Module fallback (used when no Using refs are wired).
func (s *heavyPathSelector) Forward(_ ...*autograd.Variable) *autograd.Variable {
	t, _ := tensor.FromFloat32([]float32{0}, []int64{1})
	return autograd.NewVariable(t, false) // default: lightweight path
}

// ForwardNamed receives Using refs by tag name.
func (s *heavyPathSelector) ForwardNamed(_ *autograd.Variable, refs map[string]*autograd.Variable) *autograd.Variable {
	refined := refs["refined"]

	data, _ := refined.Data().Float32Data()
	maxVal := data[0]
	for _, v := range data[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	branch := float32(0) // lightweight path
	if maxVal > 5.0 {
		branch = 1 // heavy processing path
	}

	t, _ := tensor.FromFloat32([]float32{branch}, []int64{1})
	return autograd.NewVariable(t, false)
}

func (s *heavyPathSelector) RefNames() []string          { return []string{"refined"} }
func (s *heavyPathSelector) Parameters() []*nn.Parameter { return nil }

// BuildShowcase builds the showcase graph.
//
// Data flow ([B,2] → [B,2]):
//
//	From(Linear 2→8)                              Tag("input")
//	Through(GELU) → Through(LayerNorm)
//	Split(readHead, readHead) → Mean()             multi-head read (independent params)
//	Also(Linear 8→8)                               residual: input + f(input)
//	Through(Dropout(0.1))
//	Map(readHead).Slices(4)                        shared per-position processing
//	reshape [1,8]→[2,4]                           decompose into 2 feature halves
//	Map(Linear).Each()                             Map.Each: process each half  Tag("halves")
//	Map(Linear).Over("halves")                     Map.Over: refine tagged halves
//	reshape [2,4]→[1,8]                           recompose feature vector
//	Loop(ffnBlock).For(2)                          Tag("refined"), Graph-as-Module body
//	Gate(SoftmaxRouter, Linear, Linear)             .Using("input")
//	Switch(heavyPathSelector, Linear, ffnBlock)       .Using("refined")  user-defined selector
//	Through(StateAdd()).Using("memory")              Tag("memory")  [forward ref]
//	Loop(Linear 8→8).While(ThresholdHalt(100), 5)
//	Loop(Linear 8→8).Until(ThresholdHalt(50), 7)
//	Through(Linear 8→2)
func BuildShowcase() (*graph.Graph, error) {
	const h = 8

	return graph.From(nn.MustLinear(2, h)).Tag("input"). // From + Tag
								Through(nn.NewGELU()).                                                                    // activation
								Through(nn.MustLayerNorm(h)).                                                             // normalization
								Split(readHead(h), readHead(h)).Merge(graph.Mean()).                                      // Split + Merge (multi-head read)
								Also(nn.MustLinear(h, h)).                                                                // Also (residual)
								Through(nn.NewDropout(0.1)).                                                              // regularization
								Map(readHead(2)).Slices(h/2).                                                             // Map.Slices: per-position processing
								Through(graph.Reshape(2, h/2)).                                                           // [1,8] → [2,4]: two feature halves
								Map(nn.MustLinear(h/2, h/2)).Each().Tag("halves").                                        // Map.Each: process each half
								Map(nn.MustLinear(h/2, h/2)).Over("halves").                                              // Map.Over: refine tagged halves
								Through(graph.Reshape(1, h)).                                                             // [2,4] → [1,8]: recompose
								Loop(ffnBlock(h)).For(2).Tag("refined").                                                  // Loop.For (Graph-as-Module body)
								Gate(graph.SoftmaxRouter(h, 2), nn.MustLinear(h, h), nn.MustLinear(h, h)).Using("input"). // Gate + backward ref
								Switch(&heavyPathSelector{}, nn.MustLinear(h, h), ffnBlock(h)).Using("refined").          // Switch + user-defined selector
								Through(graph.StateAdd()).Using("memory").Tag("memory").                                  // recurrent: reads previous output, tags new output
								Loop(nn.MustLinear(h, h)).While(graph.ThresholdHalt(100), 5).                             // Loop.While
								Loop(nn.MustLinear(h, h)).Until(graph.ThresholdHalt(50), 7).                              // Loop.Until
								Through(nn.MustLinear(h, 2)).                                                             // output projection
								Build()
}
