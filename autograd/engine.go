package autograd

import (
	"fmt"

	"github.com/fab2s/goDl/tensor"
)

// Backward computes gradients for all leaf variables that contributed
// to this variable's value.
//
// The variable must be a scalar (numel=1). For non-scalar outputs,
// use BackwardWithGrad to provide an external gradient.
func (v *Variable) Backward() error {
	if err := v.Err(); err != nil {
		return err
	}
	if v.data.Numel() != 1 {
		return fmt.Errorf("autograd: backward requires scalar output (got numel=%d); use BackwardWithGrad for non-scalar", v.data.Numel())
	}
	// dL/dL = 1
	// For a scalar tensor, shape may be [] (0-dim). Use shape [1] for the seed.
	shape := v.data.Shape()
	if len(shape) == 0 {
		shape = []int64{1}
	}
	seed, err := tensor.Ones(shape, tensor.WithDType(v.data.DType()), tensor.WithDevice(v.data.Device()))
	if err != nil {
		return fmt.Errorf("autograd: creating gradient seed: %w", err)
	}
	return v.BackwardWithGrad(seed)
}

// BackwardWithGrad computes gradients using the provided gradient tensor
// as the starting point. This is needed when the output is not a scalar.
func (v *Variable) BackwardWithGrad(gradOutput *tensor.Tensor) error {
	if err := v.Err(); err != nil {
		return err
	}

	// Topological sort: collect all nodes from v back to leaves.
	order := topoSort(v)

	// Map from variable pointer to accumulated gradient.
	grads := make(map[*Variable]*tensor.Tensor)
	grads[v] = gradOutput

	// Walk in reverse topological order.
	for i := len(order) - 1; i >= 0; i-- {
		node := order[i]
		grad, ok := grads[node]
		if !ok || grad == nil {
			continue
		}

		// If this is a leaf that requires grad, store the gradient.
		if node.isLeaf && node.requiresGrad {
			node.accumulateGrad(grad)
		}

		// If this node has a gradient function, propagate backward.
		if node.gradFn != nil {
			inputGrads := node.gradFn.apply(grad)
			for j, input := range node.gradFn.inputs {
				if j >= len(inputGrads) || inputGrads[j] == nil {
					continue
				}
				if !input.requiresGrad {
					continue
				}
				if existing, ok := grads[input]; ok {
					// Accumulate gradients when a variable is used multiple times.
					acc := existing.Add(inputGrads[j])
					if err := acc.Err(); err != nil {
						return fmt.Errorf("autograd: accumulating gradient at %s: %w", node.gradFn.name, err)
					}
					grads[input] = acc
				} else {
					grads[input] = inputGrads[j]
				}
			}
		}

		// Retain gradient for non-leaf only if explicitly requested.
		if !node.isLeaf && node.retainGrad {
			node.grad = grad
		}

		// Release the backward graph as we walk it. After processing a
		// node, its gradFn (closure + captured tensors) is never used
		// again. Nil it out so the captured tensors become GC-eligible
		// immediately, rather than staying alive until the user's loss
		// variable is collected.
		//
		// We keep node.data intact because the user may read it after
		// backward (e.g., loss.Item()). The data tensors still become
		// collectible sooner because the gradFn chain linking all nodes
		// together is now broken — each non-leaf becomes an isolated
		// struct rather than part of a deep reference chain.
		//
		// See docs/design/memory-management.md for the full analysis.
		if !node.isLeaf {
			node.gradFn = nil
		}

		// Drop the gradient for this intermediate node from the map —
		// it has been distributed to its inputs and is no longer needed.
		if !node.isLeaf {
			delete(grads, node)
		}
	}

	return nil
}

// accumulateGrad adds a gradient to a leaf variable's accumulated gradient.
func (v *Variable) accumulateGrad(grad *tensor.Tensor) {
	if v.grad == nil {
		v.grad = grad
	} else {
		v.grad = v.grad.Add(grad)
	}
}

// topoSort returns variables in topological order (leaves first, output last).
// Uses Kahn's algorithm on the gradient graph.
func topoSort(root *Variable) []*Variable {
	// DFS-based topological sort (post-order).
	visited := make(map[*Variable]bool)
	var order []*Variable

	var visit func(v *Variable)
	visit = func(v *Variable) {
		if visited[v] {
			return
		}
		visited[v] = true

		if v.gradFn != nil {
			for _, input := range v.gradFn.inputs {
				visit(input)
			}
		}

		order = append(order, v)
	}

	visit(root)
	return order
}
