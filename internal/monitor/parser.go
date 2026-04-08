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
		`prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\([^,]+,\s*([\d.]+)\s*tokens(?:/s| per second)\)`)
	// reGenEval matches "eval time" lines; checked after rePromptEval so
	// "prompt eval time" lines are never reached here.
	reGenEval = regexp.MustCompile(
		`\beval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\([^,]+,\s*([\d.]+)\s*tokens(?:/s| per second)\)`)
	reSlotProc  = regexp.MustCompile(`slot\s+(?:(\d+)\s+is\s+processing|launch_slot_:\s+id\s+(\d+)\s+\|.*\|\s+processing task)`)
	reSlotRel   = regexp.MustCompile(`slot\s+(?:(\d+)\s+released|release:\s+id\s+(\d+)\s+\|.*\|\s+stop processing)`)
	reModelLoad = regexp.MustCompile(`load_tensors:\s+offloading\s+(\d+)\s+(?:repeating\s+)?layers\s+to\s+GPU`)
	reVRAMAlloc = regexp.MustCompile(`load_tensors:\s+\S+\s+(?:model\s+)?buffer\s+size\s*=\s*([\d.]+)\s*MiB`)
	reListen    = regexp.MustCompile(`server is listening on http://[\d.]+:(\d+)`)
)

// EventParser implements io.Writer. It buffers bytes, splits on newlines,
// writes each complete line to slog.Debug (preserving existing behaviour), and
// sends a structured ServerEvent to ch. Sends are non-blocking; if ch is full
// the event is dropped and Dropped is incremented.
type EventParser struct {
	log     *slog.Logger
	name    string
	ch      chan<- ServerEvent
	buf     bytes.Buffer
	Dropped uint64

	// PortCh receives the listening port parsed from llama-server stdout.
	// Sent at most once; buffered so the parser never blocks.
	PortCh chan int
}

// NewEventParser returns an EventParser for the named backend.
func NewEventParser(log *slog.Logger, name string, ch chan<- ServerEvent) *EventParser {
	return &EventParser{log: log, name: name, ch: ch, PortCh: make(chan int, 1)}
}

func (p *EventParser) Write(data []byte) (int, error) {
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

func (p *EventParser) parse(line string) ServerEvent {
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
		ev.Slot, _ = strconv.Atoi(firstNonEmpty(m[1], m[2]))
		return ev
	}

	if m := reSlotRel.FindStringSubmatch(line); m != nil {
		ev.Kind = EventSlotRelease
		ev.Slot, _ = strconv.Atoi(firstNonEmpty(m[1], m[2]))
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

	if m := reListen.FindStringSubmatch(line); m != nil {
		if port, err := strconv.Atoi(m[1]); err == nil {
			select {
			case p.PortCh <- port:
			default:
			}
		}
		ev.Kind = EventRaw
		ev.Message = line
		return ev
	}

	ev.Kind = EventRaw
	ev.Message = line
	return ev
}

// firstNonEmpty returns the first non-empty string from its arguments.
func firstNonEmpty(ss ...string) string {
	for _, s := range ss {
		if s != "" {
			return s
		}
	}
	return ""
}
