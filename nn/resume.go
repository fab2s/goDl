package nn

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"

	"github.com/fab2s/goDl/tensor"
)

// Stateful can save and load its mutable internal state for training
// resume. All built-in optimizers, schedulers, and GradScaler implement
// this interface.
//
// SaveState writes only the mutable state (buffers, counters), not
// configuration. The caller is responsible for recreating the object
// with matching configuration before calling LoadState.
type Stateful interface {
	SaveState(w io.Writer) error
	LoadState(r io.Reader) error
}

// --- Checkpoint ---

var ckptMagic = [4]byte{'G', 'D', 'C', 'K'}

const ckptVersion uint32 = 1

// Checkpoint manages save/resume of an entire training state: model
// parameters, optimizer buffers, scheduler counters, and the epoch
// number. Components are registered with the builder and saved/loaded
// as a single atomic file.
//
//	ckpt := nn.NewCheckpoint("checkpoints/run01").
//	    Model(model).
//	    Add("optimizer", optimizer).
//	    Add("scheduler", scheduler)
//
//	// Save periodically during training.
//	ckpt.Save(epoch)
//
//	// Resume from latest checkpoint.
//	startEpoch, err := ckpt.Load()
type Checkpoint struct {
	dir    string
	prefix string
	model  Module
	states []namedState
}

type namedState struct {
	name string
	s    Stateful
}

// NewCheckpoint creates a checkpoint manager. pathPrefix determines the
// output directory (everything before the last path separator) and filename
// prefix (everything after).
//
//	nn.NewCheckpoint("checkpoints/run01")
//	// saves to: checkpoints/run01_000042.ckpt
func NewCheckpoint(pathPrefix string) *Checkpoint {
	return &Checkpoint{
		dir:    filepath.Dir(pathPrefix),
		prefix: filepath.Base(pathPrefix),
	}
}

// Model registers the model whose parameters will be saved and loaded.
func (c *Checkpoint) Model(m Module) *Checkpoint {
	c.model = m
	return c
}

// Add registers a named stateful component (optimizer, scheduler, scaler,
// etc.). Components are saved and loaded in registration order; names and
// order must match between save and load.
func (c *Checkpoint) Add(name string, s Stateful) *Checkpoint {
	c.states = append(c.states, namedState{name: name, s: s})
	return c
}

// Save writes a checkpoint for the given epoch.
func (c *Checkpoint) Save(epoch int) error {
	if err := os.MkdirAll(c.dir, 0o755); err != nil {
		return fmt.Errorf("create checkpoint dir: %w", err)
	}

	f, err := os.Create(c.path(epoch))
	if err != nil {
		return err
	}
	defer f.Close()

	return c.writeTo(f, epoch)
}

// Load finds the most recent checkpoint and restores all state.
// Returns the epoch of the loaded checkpoint.
func (c *Checkpoint) Load() (int, error) {
	path, err := c.latest()
	if err != nil {
		return 0, err
	}
	return c.loadFrom(path)
}

// LoadEpoch loads a specific epoch's checkpoint.
func (c *Checkpoint) LoadEpoch(epoch int) error {
	_, err := c.loadFrom(c.path(epoch))
	return err
}

func (c *Checkpoint) writeTo(w io.Writer, epoch int) error {
	// Header.
	if err := binary.Write(w, binary.LittleEndian, ckptMagic); err != nil {
		return fmt.Errorf("write magic: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, ckptVersion); err != nil {
		return fmt.Errorf("write version: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, int64(epoch)); err != nil {
		return fmt.Errorf("write epoch: %w", err)
	}

	// Model flag + parameters.
	hasModel := c.model != nil
	if err := binary.Write(w, binary.LittleEndian, hasModel); err != nil {
		return fmt.Errorf("write model flag: %w", err)
	}
	if hasModel {
		if err := SaveParameters(w, c.model.Parameters()); err != nil {
			return fmt.Errorf("save parameters: %w", err)
		}
	}

	// Stateful components.
	if err := binary.Write(w, binary.LittleEndian, uint32(len(c.states))); err != nil { //nolint:gosec // count won't overflow uint32
		return fmt.Errorf("write state count: %w", err)
	}
	for _, ns := range c.states {
		if err := writeLenString(w, ns.name); err != nil {
			return fmt.Errorf("state %q: write name: %w", ns.name, err)
		}
		if err := ns.s.SaveState(w); err != nil {
			return fmt.Errorf("state %q: save: %w", ns.name, err)
		}
	}

	return nil
}

func (c *Checkpoint) loadFrom(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()

	// Header.
	var magic [4]byte
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return 0, fmt.Errorf("read magic: %w", err)
	}
	if magic != ckptMagic {
		return 0, fmt.Errorf("invalid checkpoint: bad magic %q", magic)
	}

	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return 0, fmt.Errorf("read version: %w", err)
	}
	if version != ckptVersion {
		return 0, fmt.Errorf("unsupported checkpoint version %d (want %d)", version, ckptVersion)
	}

	var epoch int64
	if err := binary.Read(f, binary.LittleEndian, &epoch); err != nil {
		return 0, fmt.Errorf("read epoch: %w", err)
	}

	// Model flag + parameters.
	var hasModel bool
	if err := binary.Read(f, binary.LittleEndian, &hasModel); err != nil {
		return 0, fmt.Errorf("read model flag: %w", err)
	}
	if hasModel != (c.model != nil) {
		if hasModel {
			return 0, fmt.Errorf("checkpoint contains model parameters but no model registered")
		}
		return 0, fmt.Errorf("model registered but checkpoint contains no parameters")
	}
	if hasModel {
		if err := LoadParameters(f, c.model.Parameters()); err != nil {
			return 0, fmt.Errorf("load parameters: %w", err)
		}
	}

	// Stateful components.
	var count uint32
	if err := binary.Read(f, binary.LittleEndian, &count); err != nil {
		return 0, fmt.Errorf("read state count: %w", err)
	}
	if int(count) != len(c.states) {
		return 0, fmt.Errorf("state count mismatch: checkpoint has %d, registered %d", count, len(c.states))
	}
	for i, ns := range c.states {
		name, err := readLenString(f)
		if err != nil {
			return 0, fmt.Errorf("state %d: read name: %w", i, err)
		}
		if name != ns.name {
			return 0, fmt.Errorf("state %d: name mismatch: checkpoint=%q registered=%q", i, name, ns.name)
		}
		if err := ns.s.LoadState(f); err != nil {
			return 0, fmt.Errorf("state %q: load: %w", name, err)
		}
	}

	return int(epoch), nil
}

func (c *Checkpoint) path(epoch int) string {
	return filepath.Join(c.dir, fmt.Sprintf("%s_%06d.ckpt", c.prefix, epoch))
}

func (c *Checkpoint) latest() (string, error) {
	pattern := filepath.Join(c.dir, c.prefix+"_*.ckpt")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return "", err
	}
	if len(matches) == 0 {
		return "", fmt.Errorf("no checkpoints found matching %s", pattern)
	}
	sort.Strings(matches)
	return matches[len(matches)-1], nil
}

// --- State serialization helpers ---

// writeTensorState writes a tensor that may be nil (for optimizer buffers
// that haven't been initialized yet).
func writeTensorState(w io.Writer, t *tensor.Tensor) error {
	if t == nil {
		return binary.Write(w, binary.LittleEndian, uint8(0))
	}
	if err := binary.Write(w, binary.LittleEndian, uint8(1)); err != nil {
		return err
	}

	shape := t.Shape()
	if err := binary.Write(w, binary.LittleEndian, uint32(len(shape))); err != nil { //nolint:gosec // ndim won't overflow uint32
		return err
	}
	for _, s := range shape {
		if err := binary.Write(w, binary.LittleEndian, s); err != nil {
			return err
		}
	}

	data, err := t.Float32Data()
	if err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint64(len(data))); err != nil {
		return err
	}
	return binary.Write(w, binary.LittleEndian, data)
}

// readTensorState reads a tensor written by writeTensorState.
// Returns nil for a tensor that was nil when saved.
func readTensorState(r io.Reader) (*tensor.Tensor, error) {
	var present uint8
	if err := binary.Read(r, binary.LittleEndian, &present); err != nil {
		return nil, err
	}
	if present == 0 {
		return nil, nil
	}

	var ndim uint32
	if err := binary.Read(r, binary.LittleEndian, &ndim); err != nil {
		return nil, err
	}
	shape := make([]int64, ndim)
	for i := range shape {
		if err := binary.Read(r, binary.LittleEndian, &shape[i]); err != nil {
			return nil, err
		}
	}

	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	data := make([]float32, count)
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		return nil, err
	}
	return tensor.FromFloat32(data, shape)
}

func writeLenString(w io.Writer, s string) error {
	b := []byte(s)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(b))); err != nil { //nolint:gosec // string length won't overflow uint32
		return err
	}
	_, err := w.Write(b)
	return err
}

func readLenString(r io.Reader) (string, error) {
	var n uint32
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r, b); err != nil {
		return "", err
	}
	return string(b), nil
}
