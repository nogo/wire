package pidfile

import (
	"errors"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

// Path returns the path to the PID file.
func Path() string {
	if dir := os.Getenv("XDG_RUNTIME_DIR"); dir != "" {
		return filepath.Join(dir, "wire.pid")
	}
	return "/tmp/wire.pid"
}

// Write creates the PID file exclusively, removing stale files where
// the recorded process is no longer alive.
func Write(path string) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
	if err != nil {
		if !errors.Is(err, os.ErrExist) {
			return fmt.Errorf("creating PID file %s: %w", path, err)
		}
		// File exists: check whether the recorded process is alive.
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return fmt.Errorf("reading PID file: %w", readErr)
		}
		pid, parseErr := strconv.Atoi(strings.TrimSpace(string(data)))
		if parseErr == nil {
			proc, _ := os.FindProcess(pid)
			if proc.Signal(syscall.Signal(0)) == nil {
				return fmt.Errorf("wire is already running (pid %d)", pid)
			}
		}
		// Stale file: remove and retry once.
		slog.Info("removing stale PID file", "path", path)
		if removeErr := os.Remove(path); removeErr != nil {
			return fmt.Errorf("removing stale PID file: %w", removeErr)
		}
		f, err = os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0644)
		if err != nil {
			return fmt.Errorf("creating PID file after stale removal: %w", err)
		}
	}
	defer f.Close()
	_, err = fmt.Fprintf(f, "%d\n", os.Getpid())
	return err
}

// Read reads the PID from the PID file.
func Read(path string) (int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return 0, fmt.Errorf("wire is not running (no PID file at %s)", path)
		}
		return 0, fmt.Errorf("reading PID file: %w", err)
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil {
		return 0, fmt.Errorf("invalid PID file: %w", err)
	}
	return pid, nil
}
