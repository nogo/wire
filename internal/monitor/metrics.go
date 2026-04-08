package monitor

import (
	"bufio"
	"encoding/json"
	"strconv"
	"strings"
)

// ParsePrometheusGauges parses gauge values from Prometheus text format.
// Only scalar gauge lines are extracted; labels are stripped from metric names.
// Returns map[metric_name]float64.
func ParsePrometheusGauges(text string) map[string]float64 {
	result := make(map[string]float64)
	scanner := bufio.NewScanner(strings.NewReader(text))
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// Format: metric_name[{labels}] value [timestamp]
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		name := fields[0]
		if i := strings.IndexByte(name, '{'); i >= 0 {
			name = name[:i]
		}
		val, err := strconv.ParseFloat(fields[1], 64)
		if err != nil {
			continue
		}
		result[name] = val
	}
	return result
}

// SlotState is the run state of a llama-server slot.
type SlotState int

const (
	SlotIdle       SlotState = 0
	SlotProcessing SlotState = 1
)

// Slot is one element from the GET /slots JSON response.
type Slot struct {
	ID              int       `json:"-"`
	State           SlotState `json:"-"`
	NCtx            int       `json:"n_ctx"`
	TokensPredicted int       `json:"-"` // n_decoded from next_token
}

// slotJSON is the raw JSON shape from llama-server /slots.
// The format changed: state (int) → is_processing (bool),
// tokens_predicted/tokens_evaluated → next_token[].n_decoded.
type slotJSON struct {
	ID           int  `json:"id"`
	NCtx         int  `json:"n_ctx"`
	IsProcessing bool `json:"is_processing"`

	// New format: n_decoded lives inside next_token array.
	NextToken []struct {
		NDecoded int `json:"n_decoded"`
	} `json:"next_token"`
}

// ParseSlots decodes a /slots JSON response.
func ParseSlots(data []byte) ([]Slot, error) {
	var raw []slotJSON
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}
	slots := make([]Slot, len(raw))
	for i, r := range raw {
		state := SlotIdle
		if r.IsProcessing {
			state = SlotProcessing
		}
		decoded := 0
		if len(r.NextToken) > 0 {
			decoded = r.NextToken[0].NDecoded
		}
		slots[i] = Slot{
			ID:              r.ID,
			State:           state,
			NCtx:            r.NCtx,
			TokensPredicted: decoded,
		}
	}
	return slots, nil
}

// ParseProps extracts the build_info string from a /props JSON response.
func ParseProps(data []byte) (buildInfo string, err error) {
	var raw struct {
		BuildInfo string `json:"build_info"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return "", err
	}
	return raw.BuildInfo, nil
}
