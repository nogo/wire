package build

import (
	"bufio"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// Options configures a build run.
type Options struct {
	Source    string // path to llama.cpp git checkout
	GPUTarget string // AMD GPU target, e.g. gfx1151
	Prefix    string // install prefix, e.g. ~/.local
}

// DefaultOptions returns Options populated from environment variables and
// home-directory defaults.
func DefaultOptions() (Options, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return Options{}, fmt.Errorf("resolving home directory: %w", err)
	}

	src := os.Getenv("WIRE_LLAMA_SRC")
	if src == "" {
		src = filepath.Join(home, "Development", "llama.cpp")
	}

	target := os.Getenv("WIRE_GPU_TARGET")
	if target == "" {
		target = "gfx1151"
	}

	prefix := os.Getenv("WIRE_PREFIX")
	if prefix == "" {
		prefix = filepath.Join(home, ".local")
	}

	return Options{Source: src, GPUTarget: target, Prefix: prefix}, nil
}

// Build compiles and installs llama-server from the llama.cpp source tree.
func Build(opts Options) error {
	// Prerequisite check.
	if _, err := exec.LookPath("cmake"); err != nil {
		return fmt.Errorf("cmake is required: sudo pacman -S cmake (or equivalent for your distro)")
	}

	slog.Info("build: source directory", "path", opts.Source)

	// 1. Fetch latest tags.
	slog.Info("build: fetching tags")
	if err := run(opts.Source, "git", "fetch", "--tags"); err != nil {
		return fmt.Errorf("git fetch --tags: %w", err)
	}

	// 2. Find latest tag.
	latest, err := latestTag(opts.Source)
	if err != nil {
		return err
	}
	slog.Info("build: latest tag", "tag", latest)

	// 3. Check current checkout; switch only if needed.
	current, err := currentTag(opts.Source)
	if err != nil {
		return err
	}
	if current != latest {
		slog.Info("build: checking out tag", "from", current, "to", latest)
		if err := run(opts.Source, "git", "checkout", latest); err != nil {
			return fmt.Errorf("git checkout %s: %w", latest, err)
		}
	} else {
		slog.Info("build: already at latest tag", "tag", latest)
	}

	// 4. Remove previous installation if manifest exists.
	if err := Remove(opts); err != nil {
		slog.Warn("build: remove previous install", "error", err)
	}

	// 5. CMake configure.
	slog.Info("build: cmake configure")
	configureArgs := []string{
		"-B", "build",
		"-DCMAKE_BUILD_TYPE=Release",
		fmt.Sprintf("-DCMAKE_INSTALL_PREFIX=%s", opts.Prefix),
		"-DGGML_HIP=ON",
		fmt.Sprintf("-DAMDGPU_TARGETS=%s", opts.GPUTarget),
		"-DGGML_HIP_ROCWMMA_FATTN=ON",
		"-DGGML_NATIVE=ON",
	}
	if err := run(opts.Source, "cmake", configureArgs...); err != nil {
		return fmt.Errorf("cmake configure: %w", err)
	}

	// 6. CMake build.
	slog.Info("build: cmake build", "jobs", runtime.NumCPU())
	buildArgs := []string{
		"--build", "build",
		"--config", "Release",
		"-j", fmt.Sprintf("%d", runtime.NumCPU()),
	}
	if err := run(opts.Source, "cmake", buildArgs...); err != nil {
		return fmt.Errorf("cmake build: %w", err)
	}

	// 7. CMake install.
	slog.Info("build: cmake install")
	if err := run(opts.Source, "cmake", "--install", "build"); err != nil {
		return fmt.Errorf("cmake install: %w", err)
	}

	installed := filepath.Join(opts.Prefix, "bin", "llama-server")
	slog.Info("build: done", "version", latest, "path", installed)
	fmt.Printf("installed llama-server %s → %s\n", latest, installed)
	return nil
}

// Remove deletes files listed in the install manifest from a previous build.
func Remove(opts Options) error {
	manifest := filepath.Join(opts.Source, "build", "install_manifest.txt")
	f, err := os.Open(manifest)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // nothing to remove
		}
		return fmt.Errorf("opening install manifest: %w", err)
	}
	defer f.Close()

	var removed int
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		path := strings.TrimSpace(scanner.Text())
		if path == "" {
			continue
		}
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			slog.Warn("remove: could not delete file", "path", path, "error", err)
			continue
		}
		removed++
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("reading install manifest: %w", err)
	}
	fmt.Printf("removed %d installed files\n", removed)
	return nil
}

// run executes a command in dir, streaming stdout/stderr to the terminal.
func run(dir, name string, args ...string) error {
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

// latestTag returns the most recent semver-like tag reachable from HEAD.
func latestTag(dir string) (string, error) {
	cmd := exec.Command("git", "describe", "--tags", "--abbrev=0",
		"--match", "b[0-9]*")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		// Fall back: sort all tags and take the last one.
		return latestTagFallback(dir)
	}
	return strings.TrimSpace(string(out)), nil
}

func latestTagFallback(dir string) (string, error) {
	cmd := exec.Command("git", "tag", "--sort=version:refname")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("git tag: %w", err)
	}
	tags := strings.Fields(strings.TrimSpace(string(out)))
	if len(tags) == 0 {
		return "", fmt.Errorf("no tags found in %s", dir)
	}
	return tags[len(tags)-1], nil
}

// currentTag returns the tag that HEAD is checked out at, or the short hash.
func currentTag(dir string) (string, error) {
	cmd := exec.Command("git", "describe", "--tags", "--exact-match", "HEAD")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		// Not on a tag: return short hash so caller can compare safely.
		cmd2 := exec.Command("git", "rev-parse", "--short", "HEAD")
		cmd2.Dir = dir
		out2, err2 := cmd2.Output()
		if err2 != nil {
			return "", fmt.Errorf("git rev-parse: %w", err2)
		}
		return strings.TrimSpace(string(out2)), nil
	}
	return strings.TrimSpace(string(out)), nil
}
