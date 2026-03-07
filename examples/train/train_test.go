// Package train demonstrates a complete training loop using the goDl stack:
// data loader → graph forward → loss → backward → optimizer step.
//
// Task: learn cumulative sum — given [a, b], predict [a, a+b].
package train

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/data"
	"github.com/fab2s/goDl/graph"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

// buildModel creates a small graph:
//
//	Linear(2,16) → GELU → LayerNorm → Also(Linear) → Linear(16,2)
func buildModel() (*graph.Graph, error) {
	return graph.From(nn.MustLinear(2, 16)).
		Through(nn.NewGELU()).
		Through(nn.MustLayerNorm(16)).
		Also(nn.MustLinear(16, 16)).
		Through(nn.MustLinear(16, 2)).
		Build()
}

// makeDataset generates n samples: input [a,b] → target [a, a+b].
// Values are drawn from [-1, 1].
func makeDataset(n int) *data.TensorDataset {
	inputs := make([]float32, n*2)
	targets := make([]float32, n*2)
	for i := range n {
		a := rand.Float32()*2 - 1 //nolint:gosec // training data, not security
		b := rand.Float32()*2 - 1 //nolint:gosec // training data, not security
		inputs[i*2] = a
		inputs[i*2+1] = b
		targets[i*2] = a
		targets[i*2+1] = a + b
	}
	inT, _ := tensor.FromFloat32(inputs, []int64{int64(n), 2})
	tgtT, _ := tensor.FromFloat32(targets, []int64{int64(n), 2})
	return data.NewTensorDataset(inT, tgtT)
}

func TestTrainLoop(t *testing.T) {
	model, err := buildModel()
	if err != nil {
		t.Fatal("build:", err)
	}

	ds := makeDataset(200)
	loader := data.NewLoader(ds, data.LoaderConfig{
		BatchSize: 20,
		Shuffle:   true,
	})
	defer loader.Close()

	optimizer := nn.NewAdam(model.Parameters(), 0.01)

	model.SetTraining(true)

	var firstLoss, lastLoss float64
	epochs := 50
	for epoch := range epochs {
		loader.Reset()
		epochLoss := 0.0
		batches := 0

		for loader.Next() {
			inT, tgtT := loader.Batch()
			input := autograd.NewVariable(inT, true)
			target := autograd.NewVariable(tgtT, false)

			// Forward.
			pred := model.Forward(input)
			if err := pred.Err(); err != nil {
				t.Fatalf("epoch %d: forward: %v", epoch, err)
			}

			// Loss.
			loss := nn.MSELoss(pred, target)
			if err := loss.Err(); err != nil {
				t.Fatalf("epoch %d: loss: %v", epoch, err)
			}

			// Backward.
			optimizer.ZeroGrad()
			if err := loss.Backward(); err != nil {
				t.Fatalf("epoch %d: backward: %v", epoch, err)
			}

			// Clip gradients to prevent explosion.
			nn.ClipGradNorm(model.Parameters(), 1.0)

			// Update.
			optimizer.Step()

			lossVal, _ := loss.Data().Float32Data()
			epochLoss += float64(lossVal[0])
			batches++
		}
		if err := loader.Err(); err != nil {
			t.Fatal("loader:", err)
		}

		avgLoss := epochLoss / float64(batches)
		if epoch == 0 {
			firstLoss = avgLoss
		}
		lastLoss = avgLoss

		if epoch%10 == 0 || epoch == epochs-1 {
			t.Logf("epoch %3d  loss=%.6f", epoch, avgLoss)
		}
	}

	// Verify loss decreased significantly.
	if lastLoss >= firstLoss*0.5 {
		t.Errorf("training did not converge: first=%.6f last=%.6f", firstLoss, lastLoss)
	}

	// Eval: test on fresh data.
	model.SetTraining(false)
	testInput, _ := tensor.FromFloat32([]float32{0.5, 0.3}, []int64{1, 2})
	pred := model.Forward(autograd.NewVariable(testInput, false))
	vals, _ := pred.Data().Float32Data()

	t.Logf("Eval: input=[0.5, 0.3] → pred=%v (want ≈ [0.5, 0.8])", vals)

	// Check predictions are in the right ballpark.
	if math.Abs(float64(vals[0])-0.5) > 0.15 {
		t.Errorf("pred[0]=%.4f, want ≈ 0.5", vals[0])
	}
	if math.Abs(float64(vals[1])-0.8) > 0.15 {
		t.Errorf("pred[1]=%.4f, want ≈ 0.8", vals[1])
	}
}
