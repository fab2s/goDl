# The Trajectory Thesis

Neural networks are trajectory generators. An input enters a high-dimensional
space, and the network's weights define a landscape that guides it along a path
through that space. Training shapes the landscape. Inference lets the input
follow its natural trajectory. Everything else is implementation detail.

This document frames the core concepts of deep learning through the trajectory
lens and connects them to goDl's architecture decisions.

---

## Trajectories, not layers

The standard mental model of a neural network is a stack of layers: input →
hidden → hidden → output. This is a useful abstraction for building software,
but it obscures what's actually happening geometrically.

Each layer transforms a point in activation space to a new point. A forward pass
is a trajectory — a sequence of positions through a high-dimensional manifold.
The weights define the vector field that determines where each point moves next.

This isn't a metaphor. Residual networks (ResNet) made it literal: the skip
connection `x + f(x)` means each layer computes a *delta* — a small step from
the current position. He et al. (2015) showed this dramatically improves
training. Chen et al. (2018) took it further with Neural ODEs, replacing
discrete residual steps with a continuous differential equation:

```
dx/dt = f(x, t, θ)
```

The forward pass becomes solving an ODE — following a continuous trajectory
through activation space. The "layers" are just discretization steps along
that trajectory.

---

## Unified vocabulary

Most DL concepts have clean trajectory interpretations:

| Standard framing | Trajectory framing |
|---|---|
| Training | Shaping the landscape so trajectories converge correctly |
| Inference | Letting an input follow its natural trajectory |
| Loss function | Measuring how far the trajectory's endpoint is from the target region |
| Gradient descent | Adjusting the landscape to pull trajectories toward targets |
| Overfitting | Trajectories that are too narrow — only work for training inputs |
| Generalization | Wide valleys — nearby inputs follow similar paths |
| Regularization | Smoothing the landscape to prevent sharp, narrow valleys |
| Attention | Dynamically choosing which dimensions matter at each trajectory step |
| Residual connections | Making trajectory steps incremental (continuous-like flow) |
| Adaptive computation | Letting the trajectory decide its own length |
| Transfer learning | A landscape from one task has valleys useful for another |
| Dropout | Randomly blocking dimensions, forcing trajectories to be robust |
| Batch normalization | Re-centering the trajectory distribution at each step |

The trajectory frame doesn't replace the math — it provides geometric intuition
for why the math works.

---

## Data-dependent control flow as trajectory branching

Fixed-architecture networks force every input through the same trajectory
length and structure. A 50-layer ResNet always takes 50 steps. A 12-head
transformer always runs 12 parallel sub-trajectories.

Adaptive architectures let the input *choose its trajectory*:

- **Adaptive depth**: iterate until confidence is high enough → variable-length
  trajectory, short for easy inputs, long for hard ones.
- **Conditional branches**: route different inputs through different sub-networks
  → trajectories fork based on the input's position in activation space.
- **Recurrent attention**: each step chooses where to look next → the trajectory
  is literally a sequence of positions in the input space.
- **Early exit**: stop when a criterion is met → the trajectory terminates when
  it reaches a confident region.

These are the architectures Python penalizes most. Every branch evaluation, every
loop iteration, every early-exit check is a Python `if` statement with ~3-5μs
overhead plus a CUDA synchronization point. The framework *discourages*
trajectory branching through performance pressure.

goDl removes that pressure. Branches are Go `if` statements (nanoseconds).
Loops are Go `for` loops. The trajectory structure is determined by the math,
not by the framework's limitations.

---

## Gradients through trajectories

Training adaptive architectures requires backpropagation through the trajectory
structure:

- **Variable-length loops**: if input A takes 3 steps and input B takes 7, the
  backward pass unrolls 3 and 7 steps respectively. The gradient signal teaches
  the network both *what to compute at each step* and *when to stop*.
- **Conditional branches**: only the taken branch receives gradients. Over many
  training samples, each branch's weights specialize for the inputs that route
  to them.
- **Parallel paths**: independent trajectories (e.g., multiple attention heads)
  get independent gradients. The heads can specialize without interference.

This is why goDl uses a Go-native autograd engine rather than libtorch's. The
graph must capture the actual trajectory — including its branches and variable
length — not a pre-traced static computation graph. Dynamic control flow and
gradient computation must coexist naturally.

---

## The selection bias in current research

If your framework makes certain trajectory structures expensive, researchers
avoid them. This isn't a conscious choice — it's selection pressure:

**Well-explored** (cheap trajectories in Python):
- Fixed-depth feedforward (ResNet, ViT)
- Single-pass attention (Transformer)
- Constant-width parallel heads

**Under-explored** (expensive trajectories in Python):
- Recurrent attention with variable fixation count
- Tree search during training (MCTS-style)
- Iterative hypothesis refinement
- Adaptive computation depth
- Multi-scale processing with feedback loops

Biological cognition is firmly in the second category. Vision involves
sequential fixations. Reasoning involves iterative refinement. Memory recall
involves variable-depth search. The architectures that most closely model
human cognition are the ones Python punishes most.

This selection bias may be steering the entire field away from architectures
that would work better for certain problems — not because the ideas are wrong,
but because the tools make them impractical to explore.

---

## Connection to goDl's architecture

goDl's layered design maps directly to the trajectory thesis:

1. **Tensor API** (Phase 1) — the coordinate system. Points in activation space
   are tensors. Operations move points.

2. **Autograd** (Phase 2) — trajectory analysis. Given a trajectory (forward
   pass), compute how changing the landscape (weights) would change where the
   trajectory ends up (gradients).

3. **Layers & Optimizers** (Phase 3) — standard landscape components. A Linear
   layer defines a linear transformation of the trajectory. An optimizer adjusts
   the landscape based on gradient analysis.

4. **Graph Engine** (Phase 4) — trajectory orchestration. This is where
   branching, looping, parallel paths, and adaptive depth become first-class
   constructs. The graph engine doesn't just execute a fixed sequence of layers
   — it *manages trajectories* through a dynamic computation structure.

The graph engine is the key differentiator. It makes trajectory branching a
composition primitive rather than an implementation challenge.

---

## Beyond single-strategy training

Current large models are trained with essentially one strategy: predict the next
token, then refine with RLHF. All knowledge — physics, poetry, reasoning,
perception — must be acquired through that single lens. This works at scale,
but it's brute force. It's like teaching someone everything through
multiple-choice tests.

### Mixture of Experts is shallow routing

Mixture of Experts (MoE) as deployed in current models (GPT-4, Mixtral) is a
step toward structured computation: a router sends each input to a subset of
expert sub-networks. But the routing is shallow — one decision at the input
boundary, and all experts share the same architecture, the same training
objective, and the same loss function. It's "which expert computes this" not
"which strategy solves this."

### Mixture of Strategies

A deeper approach: different sub-networks trained with fundamentally different
learning strategies, composed into a single system.

- A perception module trained with supervised learning on labeled data
- A reasoning module trained with reinforcement learning on reward signals
- A memory module trained with contrastive learning on similarity
- A consistency module trained to detect contradictions in intermediate results
- A meta-controller that learns when to invoke which module

Each component has its own loss function, learning rate, and update schedule.
Some are frozen (pretrained knowledge bases), others are actively learning.
Gradients flow between components where strategies should reinforce each other,
and are blocked where they shouldn't interfere.

In the trajectory frame: each module defines a different kind of landscape, and
the meta-controller learns to compose trajectories across these landscapes.
An input might start in the perception landscape, branch into reasoning, loop
through memory retrieval, pass a consistency check, and exit — all as a single
adaptive trajectory.

### Hierarchical composition

The key insight: strategies can be composed hierarchically. A trained mixture of
experts is itself a component that can be placed inside a larger graph:

```
Level 0: Individual modules (Linear, GRU, attention heads)
Level 1: Trained sub-networks (perception, reasoning, memory)
Level 2: Strategy mixtures (MoE with learned routing)
Level 3: Meta-graph that learns to compose strategy mixtures
```

Each level is trained independently, then composed. The meta-graph at level 3
doesn't need to learn perception — it learns *when to use the perception
strategy mixture versus the reasoning one*, and how to route intermediate
results between them.

This is Graph-as-Module composition: a trained graph is a Module, which is a
node in a parent graph. The same principle that lets you nest a Linear layer
inside a Transformer block lets you nest an entire trained model inside a
meta-learning graph.

### The training challenge

Multi-strategy training raises hard questions:

**Gradient interference.** When two strategies optimize different objectives,
their gradients can conflict in shared parameters. The graph engine must support
selective gradient flow — blocking, scaling, or rerouting gradients at strategy
boundaries.

**Catastrophic forgetting.** When the meta-controller trains, it must not
destroy what the sub-modules already learned. This requires freezing, elastic
weight consolidation, or explicit memory mechanisms.

**Credit assignment.** When a composed trajectory produces a good result, which
strategy deserves the credit? The meta-controller's routing decision? The
reasoning module's computation? Proper credit assignment through branching
trajectories is an open research problem.

**Curriculum design.** What order do you train the components? Bottom-up
(modules first, then composition)? Top-down (end-to-end with structure)?
Interleaved? The training curriculum itself becomes a design decision.

These are research problems, not engineering problems. But they require a
framework where multi-strategy composition is a natural primitive — not a
fragile collection of custom training loops and manual gradient hacks.

### Why this needs goDl

In Python/PyTorch, multi-strategy training is possible but painful:

- Multiple optimizers with manual `optimizer.zero_grad()` / `optimizer.step()`
  coordination
- Gradient blocking via `detach()` scattered through the code
- Custom training loops for each strategy, manually interleaved
- No native representation of "this sub-graph uses strategy A, that one uses B"
- Dynamic routing between strategies hits Python's per-op overhead at every
  branch point

In goDl's graph engine, multi-strategy composition would be structural:

- Each sub-graph carries its training context (optimizer, loss, schedule)
- Gradient flow between sub-graphs is declared in the graph topology
- The meta-controller's branching and looping are native Go — zero overhead
- Training the meta-controller is just another backward pass through a graph
  that happens to contain other trained graphs as nodes

The framework doesn't solve the research problems. But it removes the
engineering barriers that prevent researchers from exploring them.

---

## References

- He et al. (2015) — *Deep Residual Learning for Image Recognition*. Skip
  connections as incremental trajectory steps.
- Chen et al. (2018) — *Neural Ordinary Differential Equations*. Continuous-depth
  networks as ODE trajectories.
- Graves (2016) — *Adaptive Computation Time for Recurrent Neural Networks*.
  Variable-length trajectories with a halting mechanism.
- Bengio et al. (2015) — *Conditional Computation in Neural Networks*. Gating
  and routing as trajectory branching.
- Amari (1998) — *Natural Gradient Works Efficiently in Learning*. Information
  geometry — the manifold structure of parameter space.
- Shazeer et al. (2017) — *Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer*. Learned routing to expert
  sub-networks.
- Kirkpatrick et al. (2017) — *Overcoming Catastrophic Forgetting in Neural
  Networks*. Elastic weight consolidation for multi-task learning without
  destroying prior knowledge.
- Jacobs et al. (1991) — *Adaptive Mixtures of Local Experts*. The original
  mixture of experts — competitive learning between specialized modules.
