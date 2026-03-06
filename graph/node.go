package graph

import (
	"fmt"
	"strings"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
)

// nodeFunc is the internal execution contract for a graph node.
// It receives ordered inputs (one per input port) and returns
// ordered outputs (one per output port).
type nodeFunc func(inputs []*autograd.Variable) ([]*autograd.Variable, error)

// Node is a computation unit in a graph with named input/output ports.
type Node struct {
	id          string
	inputPorts  []string
	outputPorts []string
	run         nodeFunc
	params      func() []*nn.Parameter
	module      nn.Module // nil for internal nodes (add, gated_merge, state_read)
	refTarget   nn.Module // module receiving Using refs; nil means module field
}

// wrapModule adapts an nn.Module to the graph engine's internal contract.
// If the module implements nn.NamedInputModule and the node has Using refs,
// ForwardNamed is called with a named map instead of positional args.
// The node pointer is captured so that input port names (finalized after
// wireUsing) are available at execution time.
func wrapModule(m nn.Module, node *Node) nodeFunc {
	if named, ok := m.(nn.NamedInputModule); ok {
		return func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
			refs := extractRefs(node.inputPorts, inputs)
			var result *autograd.Variable
			if refs != nil {
				result = named.ForwardNamed(inputs[0], refs)
			} else {
				result = m.Forward(inputs...)
			}
			if err := result.Err(); err != nil {
				return nil, err
			}
			return []*autograd.Variable{result}, nil
		}
	}
	return func(inputs []*autograd.Variable) ([]*autograd.Variable, error) {
		result := m.Forward(inputs...)
		if err := result.Err(); err != nil {
			return nil, err
		}
		return []*autograd.Variable{result}, nil
	}
}

// extractRefs builds a tag-name → variable map from a node's input ports.
// Using refs have port names prefixed with "ref_". Returns nil if the node
// has no ref ports at all (module should use Forward instead).
//
// Nil values (forward refs on first pass) are omitted from the map.
// Modules should use the standard Go map lookup to handle this:
//
//	if state, ok := refs["memory"]; ok { ... }
func extractRefs(ports []string, inputs []*autograd.Variable) map[string]*autograd.Variable {
	hasRefs := false
	for _, port := range ports {
		if strings.HasPrefix(port, "ref_") {
			hasRefs = true
			break
		}
	}
	if !hasRefs {
		return nil
	}

	refs := make(map[string]*autograd.Variable)
	for i, port := range ports {
		if name, ok := strings.CutPrefix(port, "ref_"); ok && inputs[i] != nil {
			refs[name] = inputs[i]
		}
	}
	return refs
}

// validateRefContracts checks RefValidator contracts at build time.
// For each node whose module (or refTarget) implements nn.RefValidator,
// it verifies that the wired Using refs match RefNames() exactly.
// Modules without RefValidator are not checked — they accept any refs.
func validateRefContracts(nodes map[string]*Node) error {
	for _, node := range nodes {
		target := node.module
		if node.refTarget != nil {
			target = node.refTarget
		}
		if target == nil {
			continue
		}
		validator, ok := target.(nn.RefValidator)
		if !ok {
			continue
		}

		// Collect actual ref port names wired to this node.
		var wiredRefs []string
		for _, port := range node.inputPorts {
			if name, found := strings.CutPrefix(port, "ref_"); found {
				wiredRefs = append(wiredRefs, name)
			}
		}

		expected := validator.RefNames()

		// Check: module declares expected refs but none are wired.
		if len(wiredRefs) == 0 {
			return fmt.Errorf(
				"graph: node %q: module %T declares RefNames %v, "+
					"but no Using refs are wired; "+
					"add .Using(%s) after the node in the graph chain",
				node.id, target, expected, joinQuoted(expected))
		}

		// Check: expected refs not wired.
		wiredSet := make(map[string]bool, len(wiredRefs))
		for _, r := range wiredRefs {
			wiredSet[r] = true
		}
		for _, exp := range expected {
			if !wiredSet[exp] {
				return fmt.Errorf(
					"graph: node %q: module %T expects Using ref %q "+
						"(declared in RefNames) but it is not wired; "+
						"add .Using(%q) after the node in the graph chain",
					node.id, target, exp, exp)
			}
		}

		// Check: wired refs not expected by module.
		expectedSet := make(map[string]bool, len(expected))
		for _, e := range expected {
			expectedSet[e] = true
		}
		for _, w := range wiredRefs {
			if !expectedSet[w] {
				return fmt.Errorf(
					"graph: node %q: Using ref %q is wired but module %T "+
						"expects only %v (declared in RefNames); "+
						"check for typos in .Using() or .Tag() calls",
					node.id, w, target, expected)
			}
		}
	}
	return nil
}

// joinQuoted formats a string slice as quoted, comma-separated values.
func joinQuoted(names []string) string {
	parts := make([]string, len(names))
	for i, n := range names {
		parts[i] = fmt.Sprintf("%q", n)
	}
	return strings.Join(parts, ", ")
}
