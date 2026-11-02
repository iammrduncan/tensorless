package gpu

import (
	"context"
	"errors"
	"io"
	"reflect"
	"strings"
	"testing"

	"github.com/iammrduncan/tensorless/internal/execx"
)

const p100 = "0, Tesla P100-PCIE-16GB, 6.0, 16384, 580.95.05, GPU-1a2b3c4d-0000-1111-2222-333344445555\n"

func TestParseP100(t *testing.T) {
	gpus, err := Parse([]byte(p100))
	if err != nil {
		t.Fatal(err)
	}
	want := []GPU{{
		Index: 0, Name: "Tesla P100-PCIE-16GB", ComputeCap: "6.0", MemoryMiB: 16384,
		Driver: "580.95.05", UUID: "GPU-1a2b3c4d-0000-1111-2222-333344445555",
	}}
	if !reflect.DeepEqual(gpus, want) {
		t.Fatalf("got %+v\nwant %+v", gpus, want)
	}
	if !gpus[0].Matches("6.0", 16000) {
		t.Errorf("the 16 GB P100 must match")
	}
}

func TestParseMultipleAndOdd(t *testing.T) {
	out := strings.Join([]string{
		"0, NVIDIA GeForce RTX 3090, 8.6, 24576, 580.95.05, GPU-aaa",
		"1, Tesla P100-PCIE-12GB, 6.0, 12288, 580.95.05, GPU-bbb",
		"2, Tesla P100-PCIE-16GB, 6.0, 16280, 580.95.05, GPU-ccc",
		"3, Weird, Name, Inc, 6.0, [N/A], 580.95.05, GPU-ddd",
		"",
	}, "\n")
	gpus, err := Parse([]byte(out))
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 4 {
		t.Fatalf("got %d GPUs", len(gpus))
	}
	var matching []int
	for _, g := range gpus {
		if g.Matches("6.0", 16000) {
			matching = append(matching, g.Index)
		}
	}
	// The 3090 has the wrong compute capability, the 12 GB P100 too little
	// memory, and the last one an unknown amount.
	if !reflect.DeepEqual(matching, []int{2}) {
		t.Errorf("matching = %v, want [2]", matching)
	}
	if gpus[3].Name != "Weird, Name, Inc" || gpus[3].MemoryMiB != 0 {
		t.Errorf("odd line parsed as %+v", gpus[3])
	}
}

func TestParse12GBP100DoesNotMatch(t *testing.T) {
	gpus, err := Parse([]byte("0, Tesla P100-PCIE-12GB, 6.0, 12288, 580.95.05, GPU-x\n"))
	if err != nil {
		t.Fatal(err)
	}
	if gpus[0].Matches("6.0", 16000) {
		t.Errorf("a 12 GB P100 must not match a 16000 MiB recipe")
	}
}

func TestParseRejectsGarbage(t *testing.T) {
	for _, s := range []string{"No devices were found", "x, a, 6.0, 1, 2, 3"} {
		if _, err := Parse([]byte(s)); err == nil {
			t.Errorf("Parse(%q) should fail", s)
		}
	}
}

func TestCompareVersions(t *testing.T) {
	tests := []struct {
		a, b string
		want int
	}{
		{"580.95.05", "525.60.13", 1},
		{"525.60.13", "525.60.13", 0},
		{"525.60.9", "525.60.13", -1}, // numeric, not lexical
		{"525.60", "525.60.0", 0},
		{"525.60", "525.60.1", -1},
		{"470.256.02", "525.60.13", -1},
		{"1000.1", "999.99.99", 1},
		{"580.95.05", "580.95.5", 0},
	}
	for _, tc := range tests {
		got, err := CompareVersions(tc.a, tc.b)
		if err != nil || got != tc.want {
			t.Errorf("CompareVersions(%q, %q) = %d, %v; want %d", tc.a, tc.b, got, err, tc.want)
		}
	}
	for _, bad := range []string{"", "r580", "580..1", "580.x"} {
		if _, err := CompareVersions(bad, "1"); err == nil {
			t.Errorf("CompareVersions(%q) should fail", bad)
		}
	}
	if ok, _ := DriverAtLeast("470.256.02", "525.60.13"); ok {
		t.Errorf("470 is older than 525")
	}
	if ok, _ := DriverAtLeast("580.95.05", "525.60.13"); !ok {
		t.Errorf("580 is newer than 525")
	}
}

func TestDetect(t *testing.T) {
	f := execx.NewFake() // no nvidia-smi
	if _, err := Detect(context.Background(), f); !errors.Is(err, ErrNoNvidiaSMI) {
		t.Errorf("missing nvidia-smi: %v", err)
	}

	f = execx.NewFake("nvidia-smi")
	f.Func = func(_ context.Context, name string, args []string, stdout, _ io.Writer) error {
		if !reflect.DeepEqual(args, QueryArgs) {
			t.Errorf("args = %v", args)
		}
		_, err := io.WriteString(stdout, p100)
		return err
	}
	gpus, err := Detect(context.Background(), f)
	if err != nil || len(gpus) != 1 || gpus[0].Name != "Tesla P100-PCIE-16GB" {
		t.Errorf("Detect = %+v, %v", gpus, err)
	}

	f.Func = func(_ context.Context, _ string, _ []string, _, stderr io.Writer) error {
		io.WriteString(stderr, "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.")
		return errors.New("exit status 9")
	}
	if _, err := Detect(context.Background(), f); err == nil || !strings.Contains(err.Error(), "couldn't communicate") {
		t.Errorf("failing nvidia-smi: %v", err)
	}
}
