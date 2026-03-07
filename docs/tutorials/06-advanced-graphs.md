# Advanced Graphs

This tutorial covers the graph builder's advanced constructs: backward
and forward references, loops, gated routing, and conditional branching.

> **Prerequisites**: [The Graph Builder](05-graph-builder.md) covers the
> basics — From, Through, Build, Also, Split/Merge, Tag, Map, and
> Graph-as-Module. Everything here builds on those primitives.

## Tag and Using — backward references

Tutorial 05 introduced `Tag` for naming points in the flow. `Using`
consumes those names. When the tag appears *before* the Using call in
the builder chain, the value is wired directly — it is available in the
same forward pass with no extra machinery:

```go
g, _ := graph.From(nn.MustLinear(4, 8)).Tag("hidden").
    Through(nn.NewGELU()).
    Through(crossAttention).Using("hidden").
    Build()
```

Here `crossAttention` receives two arguments in its `Forward` call: the
stream (output of GELU) and the tagged `"hidden"` tensor. Module
`Forward` is variadic — Using refs arrive as extra arguments after the
stream input.

You can wire multiple tags at once:

```go
Through(fusion).Using("audio", "video")
```

The module receives `(stream, audio, video)`.

## Forward references — recurrent state

When `Using` appears *before* the matching `Tag`, the builder creates a
**forward reference**. The value does not exist yet during the current
forward pass, so it is carried in a state buffer between calls to
`g.Forward()`.

This is how you build recurrent connections:

```go
g, _ := graph.From(nn.MustLinear(4, 8)).
    Through(graph.StateAdd()).Using("memory").Tag("memory").
    Through(nn.MustLinear(8, 2)).
    Build()
```

Walk through what happens:

1. `Using("memory")` appears before `Tag("memory")` — the builder
   detects this automatically and creates a state buffer.
2. On the **first** `g.Forward(input)` call, the `"memory"` state is
   nil. The graph auto-fills nil state with zeros matching the stream
   shape, so `StateAdd` computes `stream + zeros = stream` (clean
   pass-through).
3. After execution, the output of the tagged node is captured into the
   state buffer.
4. On the **second** `g.Forward(input)` call, `StateAdd` receives the
   real previous output as the `"memory"` argument: `stream + prev`.

```go
// First call: "memory" state is nil, auto-zeroed
out1 := g.Forward(input1)

// Second call: previous output feeds back as "memory"
out2 := g.Forward(input2)
```

The same tag can serve as both a forward and a backward reference in the
same graph — backward consumers see the current-pass value, forward
consumers see the previous-pass value.

### Managing state

Two methods control the state buffers:

**ResetState** clears all buffers to nil. Call this when starting a new
sequence (inference) or a new episode (RL):

```go
g.ResetState()
out := g.Forward(firstInput) // state starts fresh
```

**DetachState** breaks the gradient chain on all state buffers without
clearing the values. Call this between training steps to prevent the
autograd graph from growing without bound:

```go
for _, batch := range loader {
    output := g.Forward(batch.Input)
    loss := nn.MSELoss(output, batch.Target)
    loss.Backward()
    g.DetachState() // cut gradients, keep values
    optimizer.Step()
    optimizer.ZeroGrad()
}
```

Without `DetachState`, each step's backward pass would walk through
every previous step's computation graph — memory grows linearly with
the number of steps. Detaching caps it at one step.

### Built-in state primitive

`graph.StateAdd()` is a nil-safe additive cell: it sums all non-nil
inputs. On the first pass it acts as a pass-through (zeros + stream =
stream). On subsequent passes it adds the previous state to the current
stream. This is the standard building block for simple recurrent
feedback.

## Loops

The `Loop` builder repeats a body module, feeding each iteration's
output as the next iteration's input. Three termination modes cover
different use cases.

### Fixed iteration with For

```go
g, _ := graph.From(encoder).
    Loop(refinementStep).For(5).
    Through(decoder).
    Build()
```

The body runs exactly 5 times. Each iteration builds its own
computation graph, so the backward pass unrolls automatically —
backpropagation through time (BPTT) with no special handling.

The body can be any module, including a sub-graph:

```go
block, _ := graph.From(nn.MustLinear(8, 8)).
    Through(nn.NewGELU()).
    Through(nn.MustLayerNorm(8)).
    Build()

g, _ := graph.From(encoder).
    Loop(block).For(3).  // Graph-as-Module body
    Through(decoder).
    Build()
```

### Conditional loops with While and Until

Both `While` and `Until` take a condition module and a maximum iteration
count. The condition module receives the current state and returns a
scalar. **Positive (> 0) means halt.**

**While** checks the condition *before* each body execution. If the
condition signals halt immediately, the body never runs and the input
passes through unchanged (0 to maxIter iterations):

```go
g, _ := graph.From(encoder).
    Loop(refine).While(graph.ThresholdHalt(100), 20).
    Through(decoder).
    Build()
```

**Until** runs the body first, then checks the condition. The body
always executes at least once (1 to maxIter iterations):

```go
g, _ := graph.From(encoder).
    Loop(refine).Until(graph.LearnedHalt(hidden), 20).
    Through(decoder).
    Build()
```

The stop/continue decision is non-differentiable — gradients flow
through the body iterations only. The condition module's parameters
are still included in `Parameters()` and receive `SetTraining`
propagation, so you can train them with auxiliary losses or policy
gradient methods.

### Built-in halt conditions

**ThresholdHalt(val)** — signals halt when the max element of the state
exceeds the threshold. Parameter-free, good for convergence checks:

```go
Loop(body).Until(graph.ThresholdHalt(50), 20)
```

**LearnedHalt(dim)** — a learnable linear probe that projects the state
to a scalar. The network learns when to stop iterating. This is the
Adaptive Computation Time (ACT) pattern:

```go
Loop(body).Until(graph.LearnedHalt(hiddenDim), 20)
```

`LearnedHalt` has trainable parameters — they appear in the graph's
`Parameters()` automatically.

### Loops with Using — external references

Loop bodies often need access to data that does not change between
iterations. For example, an attention loop needs the image at every
step, but the image is not the recurrent state — the hidden vector is.

`Using` after a Loop terminator (`.For`, `.While`, `.Until`) wires a
tagged value as an extra input to the body at every iteration:

```go
g, _ := graph.From(&Identity{}).Tag("image").
    Through(NewH0Init(hiddenDim)).
    Loop(attentionStep).For(nGlimpses).Using("image").Tag("attention").
    Through(decoder).
    Build()
```

How refs reach the body depends on the body's interface:

- **NamedInputModule** — the body receives `ForwardNamed(state, refs)`
  where `refs` is a `map[string]*Variable` keyed by tag name.
- **Plain Module** — refs are appended as extra positional arguments:
  `Forward(state, image)`.

This is how you build attention mechanisms, cross-attention layers, or
any loop where the body needs a constant reference without threading it
through the recurrent state.

### Resettable — auto-reset before Forward

Loop bodies with internal mutable state (fixation location, counter,
accumulator) need to be reinitialized before each forward pass.
Rather than requiring the caller to know about and call reset methods,
implement `nn.Resettable`:

```go
type Resettable interface {
    Reset(batchSize int64)
}
```

The graph calls `Reset(batchSize)` on every `Resettable` module
before execution begins. The batch size is inferred from the first
input tensor's leading dimension. No manual reset calls needed.

```go
type AttentionStep struct {
    location *autograd.Variable // mutable per-forward state
    // ...
}

func (s *AttentionStep) Reset(batchSize int64) {
    s.location = zeros(batchSize, 2)
}
```

### Traced — per-iteration trajectory collection

Loop bodies often produce side outputs at each iteration that need to
be collected — fixation locations, attention weights, confidence
scores. The graph's stream only carries the final state, but training
losses may need the full trajectory.

Implement `nn.Traced` to have the loop executor collect side outputs
automatically:

```go
type Traced interface {
    Trace() *autograd.Variable
}
```

The loop calls `Trace()` once before the first iteration (after Reset)
to capture the initial state, then once after each iteration. The
results are stored on the loop node and accessible via `g.Traces(tag)`:

```go
func (s *AttentionStep) Trace() *autograd.Variable {
    return s.location // current fixation point
}
```

After Forward, retrieve the full trajectory:

```go
g.Forward(input)
locations := g.Traces("attention") // [initial, step1, step2, ...]
// len(locations) == nGlimpses + 1
```

This eliminates the side-channel pattern where the caller holds a
reference to an internal module and reads its state after Forward.
The model struct only needs the graph — no module-specific fields.

### Composing loop interfaces

A loop body can implement any combination of these interfaces:

| Interface | Effect |
|-----------|--------|
| `Resettable` | Auto-reset before each Forward |
| `Traced` | Per-iteration side output collection |
| `NamedInputModule` | Named ref forwarding from Using |
| `RefValidator` | Build-time validation of expected refs |

They compose independently. A body implementing all four gets
auto-reset, trace collection, named refs, and build-time validation —
all without any manual wiring in the model code.

### Context-aware loops

All loop types respect `context.Context` when invoked through
`ForwardCtx`. The context is checked between iterations — if a
timeout expires or cancellation is requested, the loop exits early:

```go
ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
defer cancel()

// While loop exits if the timeout fires, even if the condition
// hasn't signaled halt and maxIter hasn't been reached.
result := g.ForwardCtx(ctx, input)
```

Until loops preserve the at-least-once guarantee — the body runs
once before the first cancellation check. See
[Training — Context-Aware Forward](04-training.md#context-aware-forward)
for full details and training patterns.

### Forward refs inside loops

When the loop body is a sub-graph that uses forward references, the
sub-graph's state persists across loop iterations. This gives you BPTT
with recurrent connections inside the loop body for free.

## Gate — soft routing

`Gate` implements mixture-of-experts style routing. A router module
produces weights, all expert modules execute on the stream, and their
outputs are combined using the router's weights:

```go
g, _ := graph.From(nn.MustLinear(4, 8)).Tag("features").
    Through(nn.NewGELU()).
    Gate(graph.SoftmaxRouter(8, 3), expertA, expertB, expertC).Using("features").
    Through(nn.MustLinear(8, 2)).
    Build()
```

Key properties:

- **All experts execute** on every forward pass. For sparse execution,
  use Switch instead.
- **The router owns normalization.** The gated merge node applies
  weights as-is (`sum_i(weights[...,i] * expert_i)`). Whether those
  weights come from softmax, sigmoid, top-k, or something custom is
  entirely up to the router.
- **Using wires to the router**, not the experts. In the example above,
  the router sees both the stream and the `"features"` tag. This lets
  the router make routing decisions based on earlier representations.

### Built-in routers

**SoftmaxRouter(dim, n)** — linear projection to n logits, then
softmax. Weights sum to 1 (standard MoE):

```go
Gate(graph.SoftmaxRouter(hidden, 3), expertA, expertB, expertC)
```

**SigmoidRouter(dim, n)** — linear projection to n logits, then
sigmoid. Each expert is gated independently between 0 and 1:

```go
Gate(graph.SigmoidRouter(hidden, 2), expertA, expertB)
```

Both routers handle variadic inputs from Using by summing them before
projection, so extra context does not change the input dimension.

### Custom routers

Any `nn.Module` works as a router. It must return a tensor with shape
`[..., numExperts]`:

```go
type myRouter struct {
    proj *nn.Linear
}

func (r *myRouter) Forward(inputs ...*autograd.Variable) *autograd.Variable {
    // inputs[0] = stream, inputs[1:] = Using refs
    combined := inputs[0]
    for _, ref := range inputs[1:] {
        combined = combined.Add(ref)
    }
    return r.proj.Forward(combined).Softmax(-1)
}

func (r *myRouter) Parameters() []*nn.Parameter {
    return r.proj.Parameters()
}
```

## Switch — hard routing

`Switch` selects a single branch to execute based on the router's
output. Only the selected branch runs — the others are skipped entirely:

```go
g, _ := graph.From(nn.MustLinear(4, 8)).Tag("features").
    Through(nn.NewGELU()).
    Switch(graph.ArgmaxSelector(8, 2), lightPath, heavyPath).Using("features").
    Through(nn.MustLinear(8, 2)).
    Build()
```

Key properties:

- **The router returns a 0-based branch index** as a 1-element tensor.
- **Selection is non-differentiable.** Gradients flow through the
  selected branch only. The router does not receive gradient through
  the selection decision.
- **Each branch gets the stream only** (not the Using refs). Using
  refs go to the router so it can make an informed selection.
- For differentiable routing where all paths contribute, use Gate.

### Built-in selectors

**FixedSelector(idx)** — always picks the same branch. Useful for
testing, ablation, or static configuration:

```go
Switch(graph.FixedSelector(0), branchA, branchB)
```

**ArgmaxSelector(dim, n)** — learnable linear projection to n logits,
picks the branch with the highest value. Parameters are included in the
graph for training with policy-gradient methods:

```go
Switch(graph.ArgmaxSelector(hidden, 3), branchA, branchB, branchC)
```

Like the Gate routers, `ArgmaxSelector` sums variadic inputs from Using
before projection.

## Real-world example: attention with Loop + Using + Traced

Here is a complete attention-loop pattern, similar to what you'd use
for visual attention or iterative refinement with trajectory tracking:

```go
// AttentionStep is the loop body. It implements:
//   - nn.Module (Forward + Parameters)
//   - nn.NamedInputModule (ForwardNamed for Using refs)
//   - nn.RefValidator (RefNames for build-time check)
//   - nn.Resettable (Reset for auto-init before Forward)
//   - nn.Traced (Trace for per-iteration location collection)
type AttentionStep struct {
    sensor  Sensor
    gru     *nn.GRUCell
    locHead *nn.Linear
    location *autograd.Variable
}

func (s *AttentionStep) Reset(batchSize int64) {
    s.location = zeros(batchSize, 2) // initial fixation at center
}

func (s *AttentionStep) Trace() *autograd.Variable {
    return s.location
}

func (s *AttentionStep) ForwardNamed(h *autograd.Variable, refs map[string]*autograd.Variable) *autograd.Variable {
    glimpse := s.sensor.Forward(refs["image"], s.location)
    newH := s.gru.Forward(glimpse, h)
    s.location = s.locHead.Forward(newH).Tanh()
    return newH
}

func (s *AttentionStep) RefNames() []string { return []string{"image"} }
```

The graph reads like the architecture diagram:

```go
g, _ := graph.From(&Identity{}).Tag("image").
    Through(NewH0Init(hiddenDim)).
    Loop(step).For(nGlimpses).Using("image").Tag("attention").
    Through(latentHead).Tag("latent").
    Split(letterHead, caseHead).TagGroup("heads").
    Merge(&SelectFirst{}).
    Build()
```

The model struct only holds the graph — no module references needed:

```go
type Model struct {
    Graph   *graph.Graph
    Decoder *Decoder
}

func (m *Model) Forward(img, caseLabel *autograd.Variable) *Result {
    m.Graph.Forward(img)  // auto-resets, auto-traces
    return &Result{
        Logits:    m.Graph.Tagged("heads_0"),
        Latent:    m.Graph.Tagged("latent"),
        Locations: m.Graph.Traces("attention"),  // [initial, step1, ..., stepN]
        Recon:     m.Decoder.Forward(m.Graph.Tagged("latent").Cat(caseLabel, 1)),
    }
}
```

No manual `Reset` call. No `Step` field leaking internals. The graph
owns the lifecycle.

## Putting it together

The showcase example (`examples/showcase/showcase.go`) uses every
builder method in a single graph. Here is a simplified version that
combines the key advanced constructs:

```go
const h = 8

// Reusable sub-graph (Graph-as-Module).
block, _ := graph.From(nn.MustLinear(h, h)).
    Through(nn.NewGELU()).
    Through(nn.MustLayerNorm(h)).
    Build()

// Main model: encode → refine → route → recurrent state → decode.
model, _ := graph.From(nn.MustLinear(4, h)).Tag("input").
    Through(nn.NewGELU()).
    Split(nn.MustLinear(h, h), nn.MustLinear(h, h)).Merge(graph.Mean()).
    Also(nn.MustLinear(h, h)).
    Loop(block).For(2).Tag("refined").
    Gate(graph.SoftmaxRouter(h, 2), nn.MustLinear(h, h), nn.MustLinear(h, h)).Using("input").
    Switch(graph.ArgmaxSelector(h, 2), nn.MustLinear(h, h), block).Using("refined").
    Through(graph.StateAdd()).Using("memory").Tag("memory").
    Loop(nn.MustLinear(h, h)).While(graph.ThresholdHalt(100), 5).
    Through(nn.MustLinear(h, 2)).
    Build()

// Train.
optimizer := nn.NewAdam(model.Parameters(), 0.001)
model.SetTraining(true)

for step := 0; step < numSteps; step++ {
    output := model.Forward(input)
    loss := nn.MSELoss(output, target)
    loss.Backward()
    model.DetachState()
    optimizer.Step()
    optimizer.ZeroGrad()
}

// Evaluate on a new sequence.
model.SetTraining(false)
model.ResetState()
output := model.Forward(testInput)
```

## Quick reference

| Construct | Builder call | Behavior |
|-----------|-------------|----------|
| Backward ref | `Tag("x")` ... `Using("x")` | Direct wire, same pass |
| Forward ref | `Using("x")` ... `Tag("x")` | State buffer, cross-pass |
| Fixed loop | `Loop(body).For(n)` | Exactly n iterations |
| While loop | `Loop(body).While(cond, max)` | 0..max, check before body |
| Until loop | `Loop(body).Until(cond, max)` | 1..max, check after body |
| Loop + ref | `Loop(body).For(n).Using("x")` | External ref at each iteration |
| Loop traces | `g.Traces("tag")` | Side outputs from `Traced` body |
| Auto-reset | implement `nn.Resettable` | Graph resets before Forward |
| Soft routing | `Gate(router, experts...)` | All execute, weighted sum |
| Hard routing | `Switch(router, branches...)` | One executes, index select |

## What's next

This covers the full builder API. See the
[Visualization](07-visualization.md) tutorial for DOT/SVG output of
your graphs.
