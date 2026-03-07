package graph

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/fab2s/goDl/nn"
)

// DOT returns a Graphviz DOT representation of the graph.
// The output can be rendered with `dot -Tsvg graph.dot -o graph.svg`
// or pasted into an online viewer like https://dreampuf.github.io/GraphvizOnline.
//
// Composite nodes (Switch, Loop.While, Loop.Until) are expanded into
// clusters showing their internal structure — branches, body, and
// condition modules.
//
//	g, _ := graph.From(encoder).Through(decoder).Build()
//	fmt.Println(g.DOT())
func (g *Graph) DOT() string {
	var b strings.Builder
	b.WriteString("digraph G {\n")
	b.WriteString("  rankdir=TB;\n")
	b.WriteString("  fontname=\"Helvetica\";\n")
	b.WriteString("  node [fontname=\"Helvetica\" fontsize=11 style=filled];\n")
	b.WriteString("  edge [fontname=\"Helvetica\" fontsize=9];\n")
	b.WriteString("  compound=true;\n") // needed for edges to/from clusters
	b.WriteString("\n")

	// Build reverse tag lookup: nodeID → []tagName.
	nodeTags := make(map[string][]string)
	for tag, nodeID := range g.tags {
		nodeTags[nodeID] = append(nodeTags[nodeID], tag)
	}

	// Identify input and output node IDs.
	inputNodes := make(map[string]bool)
	for _, ep := range g.inputs {
		inputNodes[ep.nodeID] = true
	}
	outputNodes := make(map[string]bool)
	for _, ep := range g.outputs {
		outputNodes[ep.nodeID] = true
	}

	// Pre-scan for composite nodes that should be expanded.
	// Track exit node remapping: original nodeID → virtual exit nodeID.
	exitMap := make(map[string]string)
	expanded := make(map[string]bool)

	for _, node := range g.order {
		if sc, ok := node.module.(*switchComposite); ok {
			expanded[node.id] = true
			exitID := node.id + "_out"
			exitMap[node.id] = exitID
			_ = sc // used below
		}
		// Loop composites get richer labels but stay as single nodes.
	}

	// Emit nodes grouped by execution level.
	for i, level := range g.levels {
		b.WriteString(fmt.Sprintf("  subgraph cluster_level_%d {\n", i))
		b.WriteString(fmt.Sprintf("    label=\"level %d\";\n", i))
		b.WriteString("    style=dashed; color=\"#999999\"; fontcolor=\"#999999\";\n")
		b.WriteString("    rank=same;\n")

		for _, node := range level {
			if expanded[node.id] {
				emitSwitchCluster(&b, node, nodeTags[node.id])
			} else {
				label := nodeLabel(node, nodeTags[node.id])
				shape, fill := nodeStyle(node, inputNodes[node.id], outputNodes[node.id])
				b.WriteString(fmt.Sprintf("    %q [label=%q shape=%s fillcolor=%q];\n",
					node.id, label, shape, fill))
			}
		}
		b.WriteString("  }\n\n")
	}

	// Emit edges (remap outgoing edges from expanded composites).
	for _, edge := range g.edges {
		fromID := edge.fromNode
		if exit, ok := exitMap[fromID]; ok {
			fromID = exit
		}
		style, color, elabel := edgeStyle(edge)
		attrs := fmt.Sprintf("style=%s color=%q", style, color)
		if elabel != "" {
			attrs += fmt.Sprintf(" label=%q fontcolor=%q", elabel, color)
		}
		b.WriteString(fmt.Sprintf("  %q -> %q [%s];\n",
			fromID, edge.toNode, attrs))
	}

	// Emit forward-ref state loops (dotted, from writer back to reader).
	for _, s := range g.state {
		writerID := s.writerID
		if exit, ok := exitMap[writerID]; ok {
			writerID = exit
		}
		b.WriteString(fmt.Sprintf("  %q -> %q [style=dotted color=%q label=%q fontcolor=%q constraint=false];\n",
			writerID, s.readerID, "#e67e22", "state:"+s.name, "#e67e22"))
	}

	b.WriteString("}\n")
	return b.String()
}

// SVG renders the graph as SVG using the Graphviz dot command.
// Returns the SVG content as bytes. If a path is provided, the SVG
// is also written to that file (parent directories must exist).
//
// Requires the dot binary (from Graphviz) to be installed and in PATH.
// Install: apt install graphviz (Ubuntu), brew install graphviz (macOS).
//
//	svg, _ := g.SVG()                     // just get the bytes
//	g.SVG("graph.svg")                    // write to file
//	g.SVG("docs/architecture.svg")        // write to path
func (g *Graph) SVG(path ...string) ([]byte, error) {
	if _, err := exec.LookPath("dot"); err != nil {
		return nil, fmt.Errorf("graph: SVG rendering requires Graphviz (dot); " +
			"install with: apt install graphviz (Ubuntu), brew install graphviz (macOS)")
	}

	dot := g.DOT()
	cmd := exec.Command("dot", "-Tsvg")
	cmd.Stdin = strings.NewReader(dot)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("graph: dot render failed: %w\n%s", err, stderr.String())
	}

	svg := stdout.Bytes()

	if len(path) > 0 && path[0] != "" {
		p := filepath.Clean(path[0])
		if err := os.WriteFile(p, svg, 0600); err != nil {
			return svg, fmt.Errorf("graph: write SVG to %s: %w", p, err)
		}
	}

	return svg, nil
}

// emitSwitchCluster renders a Switch node as a cluster showing the
// router and each branch as separate visual nodes.
func emitSwitchCluster(b *strings.Builder, node *Node, tags []string) {
	sc := node.module.(*switchComposite)

	routerLabel := moduleName(sc.router)
	tagStr := ""
	if len(tags) > 0 {
		tagStr = "\\n" + tagAnnotation(tags)
	}

	fmt.Fprintf(b, "    subgraph cluster_%s {\n", node.id)
	fmt.Fprintf(b, "      label=\"Switch%s\";\n", tagStr)
	b.WriteString("      style=filled; fillcolor=\"#fef9e7\"; color=\"#f5cba7\";\n")
	b.WriteString("      fontsize=10;\n")

	// Router node (entry point — keeps the original node ID for incoming edges).
	fmt.Fprintf(b, "      %q [label=%q shape=diamond fillcolor=%q];\n",
		node.id, routerLabel, "#f5cba7")

	// Branch nodes.
	for i, branch := range sc.branches {
		branchID := fmt.Sprintf("%s_b%d", node.id, i)
		branchLabel := fmt.Sprintf("[%d] %s", i, moduleName(branch))
		fill := "#eaecee"
		if _, ok := branch.(*Graph); ok {
			fill = "#d6eaf8" // sub-graph
		}
		fmt.Fprintf(b, "      %q [label=%q shape=box fillcolor=%q];\n",
			branchID, branchLabel, fill)
	}

	// Exit merge point.
	exitID := node.id + "_out"
	fmt.Fprintf(b, "      %q [label=\"\" shape=circle width=0.15 fillcolor=%q];\n",
		exitID, "#d5dbdb")

	// Internal edges: router → branches → exit.
	for i := range sc.branches {
		branchID := fmt.Sprintf("%s_b%d", node.id, i)
		fmt.Fprintf(b, "      %q -> %q [style=dashed color=%q label=%q fontcolor=%q];\n",
			node.id, branchID, "#e67e22", fmt.Sprintf("%d", i), "#e67e22")
		fmt.Fprintf(b, "      %q -> %q [color=%q];\n",
			branchID, exitID, "#7f8c8d")
	}

	b.WriteString("    }\n")
}

// nodeLabel builds a human-readable label for a node.
func nodeLabel(node *Node, tags []string) string {
	label := cleanID(node.id)

	// Mark sub-graph modules for clarity.
	if _, ok := node.module.(*Graph); ok {
		label = "Graph (sub)"
	}

	// Enhance label for loop composites.
	if lc, ok := node.module.(*loopComposite); ok {
		label += fmt.Sprintf("\nbody: %s\ncond: %s", moduleName(lc.body), moduleName(lc.cond))
	}

	// Enhance label for map composites.
	if mc, ok := node.module.(*mapComposite); ok {
		label += fmt.Sprintf("\nbody: %s", moduleName(mc.body))
	}

	// Add tag annotations.
	if len(tags) > 0 {
		label += "\n" + tagAnnotation(tags)
	}

	return label
}

// moduleName returns a clean human-readable name for a module.
func moduleName(m nn.Module) string {
	if m == nil {
		return "nil"
	}
	t := reflect.TypeOf(m)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	name := t.Name()
	if name == "" {
		name = t.String()
	}
	// Use "Graph" for sub-graph modules.
	if name == "Graph" {
		return "Graph (sub)"
	}
	return name
}

// cleanID strips the trailing _N counter from a node ID.
func cleanID(id string) string {
	for i := len(id) - 1; i >= 0; i-- {
		if id[i] == '_' {
			suffix := id[i+1:]
			allDigits := true
			for _, c := range suffix {
				if c < '0' || c > '9' {
					allDigits = false
					break
				}
			}
			if allDigits && len(suffix) > 0 {
				return id[:i]
			}
			break
		}
	}
	return id
}

// tagAnnotation formats tag names for display.
func tagAnnotation(tags []string) string {
	parts := make([]string, len(tags))
	for i, t := range tags {
		parts[i] = "#" + t
	}
	return strings.Join(parts, " ")
}

// nodeStyle returns shape and fill color based on node type.
func nodeStyle(node *Node, isInput, isOutput bool) (shape, fill string) {
	id := node.id

	switch {
	case isInput && isOutput:
		return "doubleoctagon", "#aed6f1"
	case isInput:
		return "invhouse", "#aed6f1"
	case isOutput:
		return "house", "#a9dfbf"

	case strings.HasPrefix(id, "state_read_"):
		return "diamond", "#f9e79f"
	case strings.HasPrefix(id, "add_"):
		return "circle", "#d5dbdb"
	case strings.HasPrefix(id, "gated_merge_"):
		return "circle", "#d5dbdb"

	case strings.HasPrefix(id, "map_"):
		return "parallelogram", "#a9cce3"
	case strings.HasPrefix(id, "loop_"):
		return "box3d", "#d7bde2"
	case strings.HasPrefix(id, "switch_"):
		return "diamond", "#f5cba7"

	case strings.HasPrefix(id, "Graph_"):
		return "box", "#d6eaf8"

	case strings.HasPrefix(id, "Dropout_"):
		return "box", "#fadbd8"
	case isActivation(id):
		return "ellipse", "#fdebd0"
	case isNorm(id):
		return "box", "#d5f5e3"

	default:
		return "box", "#eaecee"
	}
}

// edgeStyle returns style, color, and optional label for an edge.
func edgeStyle(edge *Edge) (style, color, label string) {
	if strings.HasPrefix(edge.toPort, "ref_") {
		refName := strings.TrimPrefix(edge.toPort, "ref_")
		return "dashed", "#2980b9", refName
	}
	if strings.HasPrefix(edge.toPort, "expert_") || edge.toPort == "weights" {
		return "solid", "#7f8c8d", ""
	}
	return "solid", "#2c3e50", ""
}

func isActivation(id string) bool {
	for _, prefix := range []string{"GELU_", "ReLU_", "SiLU_", "Sigmoid_", "Tanh_", "Softmax_"} {
		if strings.HasPrefix(id, prefix) {
			return true
		}
	}
	return false
}

func isNorm(id string) bool {
	return strings.HasPrefix(id, "LayerNorm_") || strings.HasPrefix(id, "BatchNorm_")
}
