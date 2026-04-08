package monitor

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// GPUInfo describes a discoverable AMD GPU.
type GPUInfo struct {
	Index   int    // card index from sysfs (e.g., 1 for card1)
	Name    string // product name or "GPU <index>"
	SysPath string // e.g., "/sys/class/drm/card1/device"
	hwmon   string // cached hwmon dir path; resolved lazily
}

// GPUMetrics is a point-in-time hardware telemetry snapshot.
type GPUMetrics struct {
	Index     int     `json:"index"`
	Name      string  `json:"name"`
	BusyPct   int     `json:"busy_pct"`   // 0–100, from gpu_busy_percent
	VRAMTotal uint64  `json:"vram_total"` // bytes, from mem_info_vram_total
	VRAMUsed  uint64  `json:"vram_used"`  // bytes, from mem_info_vram_used
	TempC     float64 `json:"temp_c"`     // °C, from temp1_input / 1000
	PowerW    float64 `json:"power_w"`    // W, from power1_average / 1_000_000
}

// DiscoverGPUs finds all AMD GPUs exposed via the standard sysfs DRM tree.
// Returns an empty slice when no GPU sysfs files are present (e.g. non-AMD
// system, container without device access).
func DiscoverGPUs() []GPUInfo {
	return discoverGPUs("/sys/class/drm")
}

func discoverGPUs(base string) []GPUInfo {
	pattern := filepath.Join(base, "card*/device/gpu_busy_percent")
	matches, err := filepath.Glob(pattern)
	if err != nil || len(matches) == 0 {
		return nil
	}

	gpus := make([]GPUInfo, 0, len(matches))
	for _, match := range matches {
		// match  = <base>/card<N>/device/gpu_busy_percent
		deviceDir := filepath.Dir(match)   // <base>/card<N>/device
		cardDir := filepath.Dir(deviceDir) // <base>/card<N>
		cardName := filepath.Base(cardDir) // "card<N>"

		var index int
		fmt.Sscanf(cardName, "card%d", &index) //nolint:errcheck

		gpus = append(gpus, GPUInfo{
			Index:   index,
			Name:    gpuName(deviceDir, index),
			SysPath: deviceDir,
		})
	}
	return gpus
}

// gpuName tries to derive a human-readable GPU identifier from the uevent file.
func gpuName(deviceDir string, index int) string {
	data, err := os.ReadFile(filepath.Join(deviceDir, "uevent"))
	if err != nil {
		return fmt.Sprintf("GPU %d", index)
	}
	for _, line := range strings.Split(string(data), "\n") {
		if v, ok := strings.CutPrefix(line, "PCI_SLOT_NAME="); ok {
			return v
		}
	}
	return fmt.Sprintf("GPU %d", index)
}

// ReadGPUMetrics reads current metrics for each GPU in gpus.
// If a sysfs file cannot be read, the corresponding field is zero.
func ReadGPUMetrics(gpus []GPUInfo) []GPUMetrics {
	out := make([]GPUMetrics, 0, len(gpus))
	for i := range gpus {
		out = append(out, gpus[i].read())
	}
	return out
}

func (g *GPUInfo) read() GPUMetrics {
	m := GPUMetrics{Index: g.Index, Name: g.Name}
	m.BusyPct = readSysInt(filepath.Join(g.SysPath, "gpu_busy_percent"))
	m.VRAMTotal = readSysUint64(filepath.Join(g.SysPath, "mem_info_vram_total"))
	m.VRAMUsed = readSysUint64(filepath.Join(g.SysPath, "mem_info_vram_used"))

	if hwmon := g.resolveHwmon(); hwmon != "" {
		m.TempC = float64(readSysInt64(filepath.Join(hwmon, "temp1_input"))) / 1000.0
		m.PowerW = float64(readSysInt64(filepath.Join(hwmon, "power1_average"))) / 1_000_000.0
	}
	return m
}

// resolveHwmon returns the hwmon subdirectory path, re-globbing if the cached
// path is stale (hwmon index can change across driver reloads).
func (g *GPUInfo) resolveHwmon() string {
	if g.hwmon != "" {
		if _, err := os.Stat(g.hwmon); err == nil {
			return g.hwmon
		}
		g.hwmon = "" // stale; re-resolve below
	}
	matches, err := filepath.Glob(filepath.Join(g.SysPath, "hwmon/hwmon*"))
	if err != nil || len(matches) == 0 {
		return ""
	}
	g.hwmon = matches[0]
	return g.hwmon
}

func readSysInt(path string) int {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	v, _ := strconv.Atoi(strings.TrimSpace(string(data)))
	return v
}

func readSysUint64(path string) uint64 {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
	return v
}

func readSysInt64(path string) int64 {
	data, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	v, _ := strconv.ParseInt(strings.TrimSpace(string(data)), 10, 64)
	return v
}
