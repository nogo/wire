package vram

import (
	"log/slog"

	"github.com/nogo/wire/internal/monitor"
)

// Backend abstracts GPU VRAM discovery and metrics collection.
type Backend interface {
	Discover() []monitor.GPUInfo
	ReadMetrics() []monitor.GPUMetrics
	TotalVRAM() uint64
	UsedVRAM() uint64
	IsNoop() bool
}

// SysfsBackend uses AMD sysfs paths via the monitor package.
type SysfsBackend struct {
	gpus []monitor.GPUInfo
}

// NewSysfsBackend creates a backend that reads GPU metrics from sysfs.
func NewSysfsBackend() *SysfsBackend {
	return &SysfsBackend{}
}

// Discover finds available GPUs via sysfs. Populates internal state.
func (b *SysfsBackend) Discover() []monitor.GPUInfo {
	b.gpus = monitor.DiscoverGPUs()
	if len(b.gpus) == 0 {
		slog.Debug("sysfs backend: no GPUs discovered")
	}
	return b.gpus
}

// ReadMetrics returns current VRAM/utilization metrics for discovered GPUs.
func (b *SysfsBackend) ReadMetrics() []monitor.GPUMetrics {
	return monitor.ReadGPUMetrics(b.gpus)
}

// TotalVRAM returns the sum of all discovered GPUs' total VRAM.
func (b *SysfsBackend) TotalVRAM() uint64 {
	var total uint64
	for _, m := range monitor.ReadGPUMetrics(b.gpus) {
		total += m.VRAMTotal
	}
	return total
}

// UsedVRAM returns the sum of all discovered GPUs' currently used VRAM.
func (b *SysfsBackend) UsedVRAM() uint64 {
	var used uint64
	for _, m := range monitor.ReadGPUMetrics(b.gpus) {
		used += m.VRAMUsed
	}
	return used
}

// IsNoop returns false — this backend provides real GPU data.
func (b *SysfsBackend) IsNoop() bool { return false }

// NoopBackend is used on CPU-only machines with no GPU sysfs access.
type NoopBackend struct{}

// Discover returns an empty slice — no GPUs available.
func (NoopBackend) Discover() []monitor.GPUInfo { return nil }

// ReadMetrics returns an empty slice.
func (NoopBackend) ReadMetrics() []monitor.GPUMetrics { return nil }

// TotalVRAM returns 0 — no VRAM budget to track.
func (NoopBackend) TotalVRAM() uint64 { return 0 }

// UsedVRAM returns 0.
func (NoopBackend) UsedVRAM() uint64 { return 0 }

// IsNoop returns true — CanFit always returns true.
func (NoopBackend) IsNoop() bool { return true }
