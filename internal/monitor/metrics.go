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
	ID              int       `json:"id"`
	State           SlotState `json:"state"`
	NCtx            int       `json:"n_ctx"`
	NPredict        int       `json:"n_predict"`
	Prompt          string    `json:"prompt"`
	TokensPredicted int       `json:"tokens_predicted"`
	TokensEvaluated int       `json:"tokens_evaluated"`
}

// ParseSlots decodes a /slots JSON response.
func ParseSlots(data []byte) ([]Slot, error) {
	var slots []Slot
	return slots, json.Unmarshal(data, &slots)
}
