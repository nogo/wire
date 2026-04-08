package monitor

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func newTestParser(ch chan ServerEvent) *EventParser {
	return NewEventParser(slog.Default(), "test", ch)
}

func TestParser_PromptEval(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	line := "prompt eval time =   142.50 ms /  1847 tokens (  0.08 ms per token, 12961.40 tokens/s)\n"
	p.Write([]byte(line)) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventPromptEval {
		t.Fatalf("kind: got %v, want EventPromptEval", ev.Kind)
	}
	if ev.DurationMs != 142.50 {
		t.Errorf("DurationMs: got %v, want 142.50", ev.DurationMs)
	}
	if ev.Tokens != 1847 {
		t.Errorf("Tokens: got %v, want 1847", ev.Tokens)
	}
	if ev.TokPerSec != 12961.40 {
		t.Errorf("TokPerSec: got %v, want 12961.40", ev.TokPerSec)
	}
	if ev.Slot != -1 {
		t.Errorf("Slot: got %v, want -1", ev.Slot)
	}
	if ev.Backend != "test" {
		t.Errorf("Backend: got %q, want %q", ev.Backend, "test")
	}
}

func TestParser_GenEval(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	line := "eval time =  8234.12 ms /   412 tokens ( 19.99 ms per token,    50.04 tokens/s)\n"
	p.Write([]byte(line)) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventGenEval {
		t.Fatalf("kind: got %v, want EventGenEval", ev.Kind)
	}
	if ev.DurationMs != 8234.12 {
		t.Errorf("DurationMs: got %v, want 8234.12", ev.DurationMs)
	}
	if ev.Tokens != 412 {
		t.Errorf("Tokens: got %v, want 412", ev.Tokens)
	}
	if ev.TokPerSec != 50.04 {
		t.Errorf("TokPerSec: got %v, want 50.04", ev.TokPerSec)
	}
}

// PromptEval must not be misclassified as GenEval (reGenEval matches "eval time"
// which appears in "prompt eval time").
func TestParser_PromptEvalNotGenEval(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	line := "prompt eval time =   142.50 ms /  1847 tokens (  0.08 ms per token, 12961.40 tokens/s)\n"
	p.Write([]byte(line)) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventPromptEval {
		t.Fatalf("prompt eval line misclassified: got kind %v", ev.Kind)
	}
}

func TestParser_SlotAssign(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	p.Write([]byte("slot 3 is processing task\n")) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventSlotAssign {
		t.Fatalf("kind: got %v, want EventSlotAssign", ev.Kind)
	}
	if ev.Slot != 3 {
		t.Errorf("Slot: got %v, want 3", ev.Slot)
	}
}

func TestParser_SlotRelease(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	p.Write([]byte("slot 0 released\n")) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventSlotRelease {
		t.Fatalf("kind: got %v, want EventSlotRelease", ev.Kind)
	}
	if ev.Slot != 0 {
		t.Errorf("Slot: got %v, want 0", ev.Slot)
	}
}

func TestParser_ModelLoad(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	p.Write([]byte("llm_load_tensors: offloading 48 layers to GPU\n")) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventModelLoad {
		t.Fatalf("kind: got %v, want EventModelLoad", ev.Kind)
	}
	if !strings.Contains(ev.Message, "offloading 48") {
		t.Errorf("Message: %q does not contain expected text", ev.Message)
	}
}

func TestParser_VRAMAlloc(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	p.Write([]byte("llm_load_tensors:        CUDA0 buffer size = 16588.80 MiB\n")) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventVRAMAlloc {
		t.Fatalf("kind: got %v, want EventVRAMAlloc", ev.Kind)
	}
}

func TestParser_Raw(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	p.Write([]byte("some unrecognized log line\n")) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventRaw {
		t.Fatalf("kind: got %v, want EventRaw", ev.Kind)
	}
	if ev.Message != "some unrecognized log line" {
		t.Errorf("Message: got %q", ev.Message)
	}
}

func TestParser_PartialLine(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	// Write without trailing newline — should not emit event yet.
	p.Write([]byte("partial")) //nolint:errcheck
	select {
	case <-ch:
		t.Fatal("unexpected event for partial line")
	default:
	}

	// Complete the line.
	p.Write([]byte(" line\n")) //nolint:errcheck
	select {
	case ev := <-ch:
		if ev.Message != "partial line" {
			t.Errorf("Message: got %q, want %q", ev.Message, "partial line")
		}
	case <-time.After(time.Second):
		t.Fatal("timeout waiting for event")
	}
}

func TestParser_MultipleLines(t *testing.T) {
	ch := make(chan ServerEvent, 8)
	p := newTestParser(ch)

	p.Write([]byte("line one\nline two\nline three\n")) //nolint:errcheck

	got := 0
	for {
		select {
		case <-ch:
			got++
		default:
			goto done
		}
	}
done:
	if got != 3 {
		t.Errorf("expected 3 events, got %d", got)
	}
}

func TestParser_DropOnFullChannel(t *testing.T) {
	ch := make(chan ServerEvent, 1)
	p := newTestParser(ch)

	// Fill channel with first event.
	p.Write([]byte("slot 0 released\n")) //nolint:errcheck
	// This should be dropped.
	p.Write([]byte("slot 1 released\n")) //nolint:errcheck

	if p.Dropped != 1 {
		t.Errorf("Dropped: got %d, want 1", p.Dropped)
	}
}

// New llama-server format uses "tokens per second" instead of "tokens/s".
func TestParser_PromptEval_NewFormat(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	line := "prompt eval time =     181.35 ms /    31 tokens (    5.85 ms per token,   170.94 tokens per second)\n"
	p.Write([]byte(line)) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventPromptEval {
		t.Fatalf("kind: got %v, want EventPromptEval", ev.Kind)
	}
	if ev.Tokens != 31 {
		t.Errorf("Tokens: got %v, want 31", ev.Tokens)
	}
	if ev.TokPerSec != 170.94 {
		t.Errorf("TokPerSec: got %v, want 170.94", ev.TokPerSec)
	}
}

func TestParser_GenEval_NewFormat(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	line := "       eval time =     235.85 ms /    10 tokens (   23.58 ms per token,    42.40 tokens per second)\n"
	p.Write([]byte(line)) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventGenEval {
		t.Fatalf("kind: got %v, want EventGenEval", ev.Kind)
	}
	if ev.Tokens != 10 {
		t.Errorf("Tokens: got %v, want 10", ev.Tokens)
	}
	if ev.TokPerSec != 42.40 {
		t.Errorf("TokPerSec: got %v, want 42.40", ev.TokPerSec)
	}
}

func TestParser_SlotAssign_NewFormat(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	p.Write([]byte("slot launch_slot_: id  2 | task 0 | processing task, is_child = 0\n")) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventSlotAssign {
		t.Fatalf("kind: got %v, want EventSlotAssign", ev.Kind)
	}
	if ev.Slot != 2 {
		t.Errorf("Slot: got %v, want 2", ev.Slot)
	}
}

func TestParser_SlotRelease_NewFormat(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	p.Write([]byte("slot      release: id  0 | task 0 | stop processing: n_tokens = 40, truncated = 0\n")) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventSlotRelease {
		t.Fatalf("kind: got %v, want EventSlotRelease", ev.Kind)
	}
	if ev.Slot != 0 {
		t.Errorf("Slot: got %v, want 0", ev.Slot)
	}
}

func TestParser_ModelLoad_NewFormat(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	p.Write([]byte("load_tensors: offloading 39 repeating layers to GPU\n")) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventModelLoad {
		t.Fatalf("kind: got %v, want EventModelLoad", ev.Kind)
	}
}

func TestParser_VRAMAlloc_NewFormat(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	p.Write([]byte("load_tensors:        ROCm0 model buffer size =  4735.80 MiB\n")) //nolint:errcheck

	ev := <-ch
	if ev.Kind != EventVRAMAlloc {
		t.Fatalf("kind: got %v, want EventVRAMAlloc", ev.Kind)
	}
}

// TestParser_LiveLlamaServer verifies that the parser regexes match actual
// llama-server output. This catches silent breakage when llama-server changes
// its log format. Requires llama-server on PATH and a small GGUF model cached.
// Skipped by default; run with: go test -run TestParser_LiveLlamaServer -live
func TestParser_LiveLlamaServer(t *testing.T) {
	if !testing.Verbose() {
		t.Skip("live test: run with -v to enable, requires llama-server on PATH")
	}

	model := os.Getenv("WIRE_TEST_MODEL")
	if model == "" {
		model = "unsloth/granite-4.0-h-tiny-GGUF:Q5_K_M"
	}

	if _, err := exec.LookPath("llama-server"); err != nil {
		t.Skip("llama-server not found on PATH")
	}

	ch := make(chan ServerEvent, 256)
	parser := NewEventParser(slog.Default(), "live-test", ch)

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "llama-server",
		"-hf", model,
		"--port", "0",
		"--no-webui",
		"--metrics",
		"--parallel", "1",
		"-ngl", "99",
		"-c", "2048",
	)
	cmd.Stdout = parser
	cmd.Stderr = parser

	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start llama-server: %v", err)
	}
	defer func() {
		cmd.Process.Kill() //nolint:errcheck
		cmd.Wait()         //nolint:errcheck
	}()

	// Collect events until we see the listening port.
	var port int
	select {
	case port = <-parser.PortCh:
	case <-time.After(60 * time.Second):
		t.Fatal("timed out waiting for llama-server to start")
	}
	t.Logf("llama-server listening on port %d", port)

	// Collect startup events — expect at least model_load or vram_alloc.
	seen := make(map[EventKind]bool)
	drain(ch, seen)

	if !seen[EventModelLoad] && !seen[EventVRAMAlloc] {
		t.Error("expected EventModelLoad or EventVRAMAlloc during startup, got neither")
	}

	// Wait for health.
	healthURL := fmt.Sprintf("http://localhost:%d/health", port)
	client := &http.Client{Timeout: 2 * time.Second}
	for i := 0; i < 30; i++ {
		resp, err := client.Get(healthURL)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == 200 {
				break
			}
		}
		time.Sleep(time.Second)
	}

	// Send a request to generate eval events.
	reqBody := `{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}`
	resp, err := http.Post(
		fmt.Sprintf("http://localhost:%d/v1/chat/completions", port),
		"application/json",
		strings.NewReader(reqBody),
	)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()

	// Wait for eval events to arrive.
	time.Sleep(time.Second)
	drain(ch, seen)

	// Verify request-path events were parsed.
	if !seen[EventPromptEval] {
		t.Error("EventPromptEval not seen — prompt eval regex may be stale")
	}
	if !seen[EventGenEval] {
		t.Error("EventGenEval not seen — gen eval regex may be stale")
	}

	t.Logf("event kinds seen: %v", seen)
}

func drain(ch <-chan ServerEvent, seen map[EventKind]bool) {
	for {
		select {
		case ev := <-ch:
			seen[ev.Kind] = true
		default:
			return
		}
	}
}

func TestParser_EventTimestamp(t *testing.T) {
	ch := make(chan ServerEvent, 4)
	p := newTestParser(ch)

	before := time.Now()
	p.Write([]byte("some line\n")) //nolint:errcheck
	after := time.Now()

	ev := <-ch
	if ev.Time.Before(before) || ev.Time.After(after) {
		t.Errorf("Time %v not in range [%v, %v]", ev.Time, before, after)
	}
}
