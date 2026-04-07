package monitor

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// buildFakeSysfs creates a minimal sysfs tree under root and returns the
// device directory path. Keys "temp1_input" and "power1_average" are placed
// in a hwmon0 subdirectory; all others go directly in the device directory.
func buildFakeSysfs(t *testing.T, root string, index int, values map[string]string) string {
	t.Helper()
	cardDir := filepath.Join(root, fmt.Sprintf("card%d", index))
	deviceDir := filepath.Join(cardDir, "device")
	hwmonDir := filepath.Join(deviceDir, "hwmon", "hwmon0")
	for _, dir := range []string{deviceDir, hwmonDir} {
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatalf("MkdirAll %s: %v", dir, err)
		}
	}
	for name, val := range values {
		var path string
		switch name {
		case "temp1_input", "power1_average":
			path = filepath.Join(hwmonDir, name)
		default:
			path = filepath.Join(deviceDir, name)
		}
		if err := os.WriteFile(path, []byte(val+"\n"), 0644); err != nil {
			t.Fatalf("WriteFile %s: %v", name, err)
		}
	}
	return deviceDir
}

func TestDiscoverGPUs_found(t *testing.T) {
	root := t.TempDir()
	buildFakeSysfs(t, root, 1, map[string]string{
		"gpu_busy_percent":    "42",
		"mem_info_vram_total": "8589934592",
		"mem_info_vram_used":  "1073741824",
		"temp1_input":         "65000",
		"power1_average":      "120000000",
	})

	gpus := discoverGPUs(root)
	if len(gpus) != 1 {
		t.Fatalf("expected 1 GPU, got %d", len(gpus))
	}
	if gpus[0].Index != 1 {
		t.Errorf("Index: got %d, want 1", gpus[0].Index)
	}
	if gpus[0].Name == "" {
		t.Error("Name must not be empty")
	}
}

func TestDiscoverGPUs_none(t *testing.T) {
	root := t.TempDir()
	gpus := discoverGPUs(root)
	if len(gpus) != 0 {
		t.Errorf("expected 0 GPUs, got %d", len(gpus))
	}
}

func TestDiscoverGPUs_multiple(t *testing.T) {
	root := t.TempDir()
	for _, idx := range []int{0, 1} {
		buildFakeSysfs(t, root, idx, map[string]string{
			"gpu_busy_percent":    "10",
			"mem_info_vram_total": "4294967296",
			"mem_info_vram_used":  "0",
			"temp1_input":         "40000",
			"power1_average":      "50000000",
		})
	}
	gpus := discoverGPUs(root)
	if len(gpus) != 2 {
		t.Errorf("expected 2 GPUs, got %d", len(gpus))
	}
}

func TestReadGPUMetrics_values(t *testing.T) {
	root := t.TempDir()
	deviceDir := buildFakeSysfs(t, root, 0, map[string]string{
		"gpu_busy_percent":    "75",
		"mem_info_vram_total": "17179869184", // 16 GiB
		"mem_info_vram_used":  "8589934592",  // 8 GiB
		"temp1_input":         "72500",       // 72.5 °C
		"power1_average":      "250000000",   // 250 W
	})

	gpu := GPUInfo{Index: 0, Name: "Test GPU", SysPath: deviceDir}
	metrics := ReadGPUMetrics([]GPUInfo{gpu})

	if len(metrics) != 1 {
		t.Fatalf("expected 1 metric, got %d", len(metrics))
	}
	m := metrics[0]

	if m.BusyPct != 75 {
		t.Errorf("BusyPct: got %d, want 75", m.BusyPct)
	}
	if m.VRAMTotal != 17179869184 {
		t.Errorf("VRAMTotal: got %d, want 17179869184", m.VRAMTotal)
	}
	if m.VRAMUsed != 8589934592 {
		t.Errorf("VRAMUsed: got %d, want 8589934592", m.VRAMUsed)
	}
	if m.TempC != 72.5 {
		t.Errorf("TempC: got %v, want 72.5", m.TempC)
	}
	if m.PowerW != 250.0 {
		t.Errorf("PowerW: got %v, want 250.0", m.PowerW)
	}
}

func TestReadGPUMetrics_missingFile(t *testing.T) {
	root := t.TempDir()
	// Only create gpu_busy_percent; omit everything else.
	deviceDir := buildFakeSysfs(t, root, 0, map[string]string{
		"gpu_busy_percent": "50",
	})
	// Remove hwmon dir so temp/power are missing.
	os.RemoveAll(filepath.Join(deviceDir, "hwmon")) //nolint:errcheck

	gpu := GPUInfo{Index: 0, Name: "Partial GPU", SysPath: deviceDir}
	metrics := ReadGPUMetrics([]GPUInfo{gpu})

	m := metrics[0]
	if m.BusyPct != 50 {
		t.Errorf("BusyPct: got %d, want 50", m.BusyPct)
	}
	if m.VRAMTotal != 0 {
		t.Errorf("VRAMTotal: expected 0 for missing file, got %d", m.VRAMTotal)
	}
	if m.TempC != 0 {
		t.Errorf("TempC: expected 0 for missing hwmon, got %v", m.TempC)
	}
	if m.PowerW != 0 {
		t.Errorf("PowerW: expected 0 for missing hwmon, got %v", m.PowerW)
	}
}

func TestReadGPUMetrics_hwmonResolve(t *testing.T) {
	root := t.TempDir()
	deviceDir := buildFakeSysfs(t, root, 0, map[string]string{
		"gpu_busy_percent": "10",
		"temp1_input":      "55000",
		"power1_average":   "100000000",
	})

	gpu := GPUInfo{Index: 0, Name: "GPU 0", SysPath: deviceDir}

	// First read — resolves hwmon via glob.
	m1 := ReadGPUMetrics([]GPUInfo{gpu})[0]
	if m1.TempC != 55.0 {
		t.Errorf("first read TempC: got %v, want 55.0", m1.TempC)
	}

	// Simulate hwmon index change: rename hwmon0 → hwmon1.
	oldPath := filepath.Join(deviceDir, "hwmon", "hwmon0")
	newPath := filepath.Join(deviceDir, "hwmon", "hwmon1")
	if err := os.Rename(oldPath, newPath); err != nil {
		t.Fatalf("rename hwmon: %v", err)
	}

	// Inject stale cache; resolveHwmon must re-glob.
	gpu.hwmon = oldPath
	m2 := ReadGPUMetrics([]GPUInfo{gpu})[0]
	if m2.TempC != 55.0 {
		t.Errorf("after hwmon rename TempC: got %v, want 55.0", m2.TempC)
	}
}

func TestGPUName_uevent(t *testing.T) {
	dir := t.TempDir()
	uevent := "DRIVER=amdgpu\nPCI_SLOT_NAME=0000:03:00.0\nMODALIAS=pci:foo\n"
	if err := os.WriteFile(filepath.Join(dir, "uevent"), []byte(uevent), 0644); err != nil {
		t.Fatal(err)
	}
	name := gpuName(dir, 1)
	if name != "0000:03:00.0" {
		t.Errorf("gpuName: got %q, want %q", name, "0000:03:00.0")
	}
}

func TestGPUName_fallback(t *testing.T) {
	dir := t.TempDir() // no uevent file
	name := gpuName(dir, 3)
	if name != "GPU 3" {
		t.Errorf("gpuName fallback: got %q, want %q", name, "GPU 3")
	}
}
