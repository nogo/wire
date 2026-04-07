package monitor

import (
	"log/slog"
	"strings"
	"testing"
	"time"
)

func newTestParser(ch chan ServerEvent) *eventParser {
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
