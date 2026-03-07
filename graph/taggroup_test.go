package graph

import (
	"testing"

	"github.com/fab2s/goDl/autograd"
	"github.com/fab2s/goDl/nn"
	"github.com/fab2s/goDl/tensor"
)

func makeTagGroupInput() *autograd.Variable {
	t, _ := tensor.FromFloat32([]float32{1, 2, 3, 4}, []int64{1, 4})
	return autograd.NewVariable(t, false)
}

// TestTagGroupBasic verifies TagGroup creates suffixed tags and registers the group.
func TestTagGroupBasic(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4), nn.MustLinear(4, 4)).TagGroup("head").
		Merge(Mean()).
		Through(nn.MustLinear(4, 2)).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	// Group should be registered.
	members := g.TagGroup("head")
	if len(members) != 3 {
		t.Fatalf("expected 3 members, got %d", len(members))
	}
	for i, want := range []string{"head_0", "head_1", "head_2"} {
		if members[i] != want {
			t.Fatalf("member %d: got %q, want %q", i, members[i], want)
		}
	}

	// Unknown group returns nil.
	if g.TagGroup("nonexistent") != nil {
		t.Fatal("expected nil for unknown group")
	}
}

// TestTagGroupTaggedOutputs verifies that suffixed tags capture outputs during Forward.
func TestTagGroupTaggedOutputs(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).TagGroup("branch").
		Merge(Mean()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.Forward(makeTagGroupInput())

	for _, tag := range []string{"branch_0", "branch_1"} {
		if g.Tagged(tag) == nil {
			t.Fatalf("Tagged(%q) should not be nil after Forward", tag)
		}
	}
}

// TestTagGroupObservation verifies Collect/Flush/Trend work with suffixed tags.
func TestTagGroupObservation(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 1), nn.MustLinear(4, 1)).TagGroup("head").
		Merge(Mean()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	for epoch := 0; epoch < 3; epoch++ {
		g.Forward(makeTagGroupInput())
		g.Collect("head_0", "head_1")
		g.Flush()
	}

	if g.Trend("head_0").Len() != 3 {
		t.Fatalf("expected 3 epochs for head_0, got %d", g.Trend("head_0").Len())
	}
	if g.Trend("head_1").Len() != 3 {
		t.Fatalf("expected 3 epochs for head_1, got %d", g.Trend("head_1").Len())
	}
}

// TestTagGroupSingleStreamError verifies TagGroup fails on single stream.
func TestTagGroupSingleStreamError(t *testing.T) {
	_, err := From(nn.MustLinear(4, 4)).
		TagGroup("fail").
		Through(nn.MustLinear(4, 2)).
		Build()
	if err == nil {
		t.Fatal("expected error for TagGroup on single stream")
	}
	want := "TagGroup"
	if got := err.Error(); !contains(got, want) {
		t.Fatalf("error should mention TagGroup, got: %s", got)
	}
}

// TestTagGroupDuplicateError verifies duplicate group name is caught.
func TestTagGroupDuplicateError(t *testing.T) {
	_, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).TagGroup("dup").
		Merge(Mean()).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).TagGroup("dup").
		Merge(Mean()).
		Build()
	if err == nil {
		t.Fatal("expected error for duplicate TagGroup")
	}
	if got := err.Error(); !contains(got, "duplicate tag group") {
		t.Fatalf("unexpected error: %s", got)
	}
}

// TestTagGroupConflictWithTag verifies group name collision with existing Tag.
func TestTagGroupConflictWithTag(t *testing.T) {
	_, err := From(nn.MustLinear(4, 4)).Tag("head").
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).TagGroup("head").
		Merge(Mean()).
		Build()
	if err == nil {
		t.Fatal("expected error for TagGroup conflicting with Tag")
	}
	if got := err.Error(); !contains(got, "conflicts with existing Tag") {
		t.Fatalf("unexpected error: %s", got)
	}
}

// TestTagGroupSuffixConflictWithTag verifies suffixed name collision with existing Tag.
func TestTagGroupSuffixConflictWithTag(t *testing.T) {
	_, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).Merge(Mean()).Tag("head_0").
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).TagGroup("head").
		Merge(Mean()).
		Build()
	if err == nil {
		t.Fatal("expected error for TagGroup suffix conflicting with Tag")
	}
	if got := err.Error(); !contains(got, "suffixed name") && !contains(got, "conflicts") {
		t.Fatalf("unexpected error: %s", got)
	}
}

// TestTagGroupTiming verifies timing profiling works with TagGroup.
func TestTagGroupTiming(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).TagGroup("head").
		Merge(Mean()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.EnableProfiling()
	g.Forward(makeTagGroupInput())

	if g.Timing("head_0") <= 0 {
		t.Fatal("head_0 timing should be positive")
	}
	if g.Timing("head_1") <= 0 {
		t.Fatal("head_1 timing should be positive")
	}
}

// TestTagGroupParametersByTag verifies ParametersByTag works with suffixed tags.
func TestTagGroupParametersByTag(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).TagGroup("branch").
		Merge(Mean()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	p0 := g.ParametersByTag("branch_0")
	p1 := g.ParametersByTag("branch_1")
	if len(p0) == 0 {
		t.Fatal("branch_0 should have parameters")
	}
	if len(p1) == 0 {
		t.Fatal("branch_1 should have parameters")
	}
}

// TestTagGroupFreeze verifies Freeze/Unfreeze work with suffixed tags.
func TestTagGroupFreeze(t *testing.T) {
	g, err := From(nn.MustLinear(4, 4)).
		Split(nn.MustLinear(4, 4), nn.MustLinear(4, 4)).TagGroup("branch").
		Merge(Mean()).
		Build()
	if err != nil {
		t.Fatal(err)
	}

	g.Freeze("branch_0", "branch_1")
	g.Forward(makeTagGroupInput())
	g.ZeroFrozenGrads() // should not panic
	g.Unfreeze("branch_0", "branch_1")
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && containsStr(s, sub))
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
