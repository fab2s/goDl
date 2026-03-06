package nn

import (
	"encoding/binary"
	"fmt"
	"io"

	"github.com/fab2s/goDl/tensor"
)

var checkpointMagic = [4]byte{'G', 'O', 'D', 'L'}

const checkpointVersion uint32 = 1

// SaveParameters writes all parameters to w in a binary format.
// Parameters are written in order; LoadParameters expects the same order.
//
//	f, _ := os.Create("model.bin")
//	nn.SaveParameters(f, model.Parameters())
//	f.Close()
func SaveParameters(w io.Writer, params []*Parameter) error {
	if err := binary.Write(w, binary.LittleEndian, checkpointMagic); err != nil {
		return fmt.Errorf("write magic: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, checkpointVersion); err != nil {
		return fmt.Errorf("write version: %w", err)
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(params))); err != nil { //nolint:gosec // param count won't overflow uint32
		return fmt.Errorf("write count: %w", err)
	}

	for _, p := range params {
		if err := writeParam(w, p); err != nil {
			return fmt.Errorf("parameter %q: %w", p.Name, err)
		}
	}
	return nil
}

// LoadParameters reads parameters from r and loads them into params.
// The parameter count, names, and shapes must match what was saved.
//
//	f, _ := os.Open("model.bin")
//	nn.LoadParameters(f, model.Parameters())
//	f.Close()
func LoadParameters(r io.Reader, params []*Parameter) error {
	var m [4]byte
	if err := binary.Read(r, binary.LittleEndian, &m); err != nil {
		return fmt.Errorf("read magic: %w", err)
	}
	if m != checkpointMagic {
		return fmt.Errorf("invalid checkpoint: bad magic %q", m)
	}

	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return fmt.Errorf("read version: %w", err)
	}
	if version != checkpointVersion {
		return fmt.Errorf("unsupported checkpoint version %d (want %d)", version, checkpointVersion)
	}

	var count uint32
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return fmt.Errorf("read count: %w", err)
	}
	if int(count) != len(params) {
		return fmt.Errorf("parameter count mismatch: checkpoint has %d, model has %d", count, len(params))
	}

	for i, p := range params {
		if err := readParam(r, p, i); err != nil {
			return err
		}
	}
	return nil
}

func writeParam(w io.Writer, p *Parameter) error {
	// Name.
	name := []byte(p.Name)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(name))); err != nil { //nolint:gosec // name length won't overflow uint32
		return err
	}
	if _, err := w.Write(name); err != nil {
		return err
	}

	// Shape.
	shape := p.Data().Shape()
	if err := binary.Write(w, binary.LittleEndian, uint32(len(shape))); err != nil { //nolint:gosec // ndim won't overflow uint32
		return err
	}
	for _, s := range shape {
		if err := binary.Write(w, binary.LittleEndian, s); err != nil {
			return err
		}
	}

	// Data (always float32, copied to CPU by Float32Data if needed).
	data, err := p.Data().Float32Data()
	if err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint64(len(data))); err != nil {
		return err
	}
	return binary.Write(w, binary.LittleEndian, data)
}

func readParam(r io.Reader, p *Parameter, index int) error {
	// Name.
	var nameLen uint32
	if err := binary.Read(r, binary.LittleEndian, &nameLen); err != nil {
		return fmt.Errorf("parameter %d: read name: %w", index, err)
	}
	nameBytes := make([]byte, nameLen)
	if _, err := io.ReadFull(r, nameBytes); err != nil {
		return fmt.Errorf("parameter %d: read name data: %w", index, err)
	}
	name := string(nameBytes)
	if name != p.Name {
		return fmt.Errorf("parameter %d: name mismatch: checkpoint=%q model=%q", index, name, p.Name)
	}

	// Shape.
	var ndim uint32
	if err := binary.Read(r, binary.LittleEndian, &ndim); err != nil {
		return fmt.Errorf("parameter %q: read ndim: %w", name, err)
	}
	shape := make([]int64, ndim)
	for j := range shape {
		if err := binary.Read(r, binary.LittleEndian, &shape[j]); err != nil {
			return fmt.Errorf("parameter %q: read shape: %w", name, err)
		}
	}

	// Validate shape.
	pShape := p.Data().Shape()
	if !shapesMatch(shape, pShape) {
		return fmt.Errorf("parameter %q: shape mismatch: checkpoint=%v model=%v", name, shape, pShape)
	}

	// Data.
	var numElements uint64
	if err := binary.Read(r, binary.LittleEndian, &numElements); err != nil {
		return fmt.Errorf("parameter %q: read data length: %w", name, err)
	}
	data := make([]float32, numElements)
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		return fmt.Errorf("parameter %q: read data: %w", name, err)
	}

	t, err := tensor.FromFloat32(data, shape)
	if err != nil {
		return fmt.Errorf("parameter %q: create tensor: %w", name, err)
	}
	p.SetData(t)
	return nil
}

func shapesMatch(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
