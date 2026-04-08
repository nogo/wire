package monitor

import (
	"encoding/json"
	"testing"
)

func TestParsePrometheusGauges_basic(t *testing.T) {
	text := `# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 31
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 2
# TYPE llamacpp:predicted_tokens_seconds gauge
llamacpp:predicted_tokens_seconds 50.04
`
	got := ParsePrometheusGauges(text)

	cases := []struct {
		key  string
		want float64
	}{
		{"llamacpp:prompt_tokens_total", 31},
		{"llamacpp:requests_processing", 2},
		{"llamacpp:predicted_tokens_seconds", 50.04},
	}
	for _, c := range cases {
		if got[c.key] != c.want {
			t.Errorf("%s: got %v, want %v", c.key, got[c.key], c.want)
		}
	}
}

func TestParsePrometheusGauges_labelsStripped(t *testing.T) {
	text := `metric_with_labels{job="llama",instance="localhost:8080"} 42
`
	got := ParsePrometheusGauges(text)
	if got["metric_with_labels"] != 42 {
		t.Errorf("expected 42, got %v", got["metric_with_labels"])
	}
}

func TestParsePrometheusGauges_ignoresComments(t *testing.T) {
	text := `# HELP foo description
# TYPE foo gauge
foo 1
`
	got := ParsePrometheusGauges(text)
	if _, ok := got["# HELP foo description"]; ok {
		t.Error("comment line was parsed as metric")
	}
	if got["foo"] != 1 {
		t.Errorf("foo: got %v, want 1", got["foo"])
	}
}

func TestParsePrometheusGauges_empty(t *testing.T) {
	got := ParsePrometheusGauges("")
	if len(got) != 0 {
		t.Errorf("expected empty map, got %v", got)
	}
}

func TestParsePrometheusGauges_invalidValue(t *testing.T) {
	text := `good_metric 3.14
bad_metric notanumber
`
	got := ParsePrometheusGauges(text)
	if got["good_metric"] != 3.14 {
		t.Errorf("good_metric: got %v, want 3.14", got["good_metric"])
	}
	if _, ok := got["bad_metric"]; ok {
		t.Error("bad_metric should not be in result")
	}
}

func TestParseSlots_basic(t *testing.T) {
	data, _ := json.Marshal([]map[string]any{
		{"id": 0, "n_ctx": 4096, "is_processing": true, "next_token": []map[string]any{{"n_decoded": 100}}},
		{"id": 1, "n_ctx": 4096, "is_processing": false, "next_token": []map[string]any{{"n_decoded": 0}}},
	})

	slots, err := ParseSlots(data)
	if err != nil {
		t.Fatalf("ParseSlots: %v", err)
	}
	if len(slots) != 2 {
		t.Fatalf("len: got %d, want 2", len(slots))
	}
	if slots[0].ID != 0 || slots[0].State != SlotProcessing {
		t.Errorf("slot[0]: %+v", slots[0])
	}
	if slots[1].ID != 1 || slots[1].State != SlotIdle {
		t.Errorf("slot[1]: %+v", slots[1])
	}
	if slots[0].TokensPredicted != 100 {
		t.Errorf("TokensPredicted: got %d, want 100", slots[0].TokensPredicted)
	}
}

func TestParseSlots_empty(t *testing.T) {
	slots, err := ParseSlots([]byte("[]"))
	if err != nil {
		t.Fatalf("ParseSlots: %v", err)
	}
	if len(slots) != 0 {
		t.Errorf("expected empty, got %v", slots)
	}
}

func TestParseSlots_invalid(t *testing.T) {
	_, err := ParseSlots([]byte("not json"))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}
