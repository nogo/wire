package monitor

import (
	"bytes"
	"log/slog"
	"regexp"
	"strconv"
	"time"
)

// Compiled regexes for llama-server stderr patterns.
// Initialised once at package init — never per-line.
var (
	rePromptEval = regexp.MustCompile(
		`prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\([^,]+,\s*([\d.]+)\s*tokens/s\)`)
	// reGenEval matches "eval time" lines; checked after rePromptEval so
	// "prompt eval time" lines are never reached here.
	reGenEval = regexp.MustCompile(
		`\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\([^,]+,\s*([\d.]+)\s*tokens/s\)`)
	reSlotProc  = regexp.MustCompile(`slot\s+(\d+)\s+is\s+processing`)
	reSlotRel   = regexp.MustCompile(`slot\s+(\d+)\s+released`)
	reModelLoad = regexp.MustCompile(`llm_load_tensors:\s+offloading\s+(\d+)\s+layers\s+to\s+GPU`)
	reVRAMAlloc = regexp.MustCompile(`llm_load_tensors:\s+\S+\s+buffer\s+size\s*=\s*([\d.]+)\s*MiB`)
)

// eventParser implements io.Writer. It buffers bytes, splits on newlines,
// writes each complete line to slog.Debug (preserving existing behaviour), and
// sends a structured ServerEvent to ch. Sends are non-blocking; if ch is full
// the event is dropped and Dropped is incremented.
type eventParser struct {
	log     *slog.Logger
	name    string
	ch      chan<- ServerEvent
	buf     bytes.Buffer
	Dropped uint64
}

// NewEventParser returns an eventParser for the named backend.
func NewEventParser(log *slog.Logger, name string, ch chan<- ServerEvent) *eventParser {
	return &eventParser{log: log, name: name, ch: ch}
}

func (p *eventParser) Write(data []byte) (int, error) {
	p.buf.Write(data)
	for {
		b := p.buf.Bytes()
		idx := bytes.IndexByte(b, '\n')
		if idx < 0 {
			break
		}
		line := string(b[:idx])
		p.buf.Next(idx + 1)
		if len(line) == 0 {
			continue
		}
		p.log.Debug(line, "backend", p.name)
		ev := p.parse(line)
		select {
		case p.ch <- ev:
		default:
			p.Dropped++
		}
	}
	return len(data), nil
}

func (p *eventParser) parse(line string) ServerEvent {
	ev := ServerEvent{
		Time:    time.Now(),
		Backend: p.name,
		Slot:    -1,
	}

	if m := rePromptEval.FindStringSubmatch(line); m != nil {
		ev.Kind = EventPromptEval
		ev.DurationMs, _ = strconv.ParseFloat(m[1], 64)
		ev.Tokens, _ = strconv.Atoi(m[2])
		ev.TokPerSec, _ = strconv.ParseFloat(m[3], 64)
		return ev
	}

	if m := reGenEval.FindStringSubmatch(line); m != nil {
		ev.Kind = EventGenEval
		ev.DurationMs, _ = strconv.ParseFloat(m[1], 64)
		ev.Tokens, _ = strconv.Atoi(m[2])
		ev.TokPerSec, _ = strconv.ParseFloat(m[3], 64)
		return ev
	}

	if m := reSlotProc.FindStringSubmatch(line); m != nil {
		ev.Kind = EventSlotAssign
		ev.Slot, _ = strconv.Atoi(m[1])
		return ev
	}

	if m := reSlotRel.FindStringSubmatch(line); m != nil {
		ev.Kind = EventSlotRelease
		ev.Slot, _ = strconv.Atoi(m[1])
		return ev
	}

	if reModelLoad.MatchString(line) {
		ev.Kind = EventModelLoad
		ev.Message = line
		return ev
	}

	if reVRAMAlloc.MatchString(line) {
		ev.Kind = EventVRAMAlloc
		ev.Message = line
		return ev
	}

	ev.Kind = EventRaw
	ev.Message = line
	return ev
}
