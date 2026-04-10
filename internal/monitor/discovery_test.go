package monitor

import (
	"encoding/json"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestModelDiscovery_FetchAndSnapshot(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("Authorization = %q, want Bearer test-key", r.Header.Get("Authorization"))
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"object": "list",
			"data": []map[string]string{
				{"id": "model-a", "owned_by": "org-a"},
				{"id": "model-b", "owned_by": "org-b"},
			},
		})
	}))
	defer srv.Close()

	d := NewModelDiscovery([]ProviderEndpoint{{
		Name:    "test-provider",
		BaseURL: srv.URL,
		APIKey:  "test-key",
	}}, slog.Default())

	// Manual refresh (don't Start the poll loop in tests).
	d.refresh()

	snap := d.Snapshot()
	models, ok := snap.ByProvider["test-provider"]
	if !ok {
		t.Fatal("expected test-provider in snapshot")
	}
	if len(models) != 2 {
		t.Fatalf("model count = %d, want 2", len(models))
	}
	if models[0].ID != "model-a" {
		t.Errorf("models[0].ID = %q, want %q", models[0].ID, "model-a")
	}
	if models[0].Provider != "test-provider" {
		t.Errorf("models[0].Provider = %q, want %q", models[0].Provider, "test-provider")
	}
}

func TestModelDiscovery_FetchError_KeepsPrevious(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		if calls == 1 {
			json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
				"data": []map[string]string{{"id": "model-a"}},
			})
			return
		}
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	d := NewModelDiscovery([]ProviderEndpoint{{
		Name:    "prov",
		BaseURL: srv.URL,
		APIKey:  "key",
	}}, slog.Default())

	d.refresh() // first call succeeds
	snap1 := d.Snapshot()
	if len(snap1.ByProvider["prov"]) != 1 {
		t.Fatalf("after first refresh: got %d models, want 1", len(snap1.ByProvider["prov"]))
	}

	d.refresh() // second call fails — previous data retained
	snap2 := d.Snapshot()
	if len(snap2.ByProvider["prov"]) != 1 {
		t.Fatalf("after failed refresh: got %d models, want 1 (retained)", len(snap2.ByProvider["prov"]))
	}
}

func TestDiscoverySnapshot_Lookup(t *testing.T) {
	snap := DiscoverySnapshot{
		ByProvider: map[string][]DiscoveredModel{
			"beta":  {{ID: "shared-model", Provider: "beta"}},
			"alpha": {{ID: "shared-model", Provider: "alpha"}, {ID: "unique", Provider: "alpha"}},
		},
	}

	// Lookup picks alphabetically first provider for conflicts.
	dm, ok := snap.Lookup("shared-model")
	if !ok {
		t.Fatal("expected to find shared-model")
	}
	if dm.Provider != "alpha" {
		t.Errorf("provider = %q, want %q (alphabetically first)", dm.Provider, "alpha")
	}

	dm, ok = snap.Lookup("unique")
	if !ok {
		t.Fatal("expected to find unique")
	}
	if dm.Provider != "alpha" {
		t.Errorf("provider = %q, want %q", dm.Provider, "alpha")
	}

	_, ok = snap.Lookup("nonexistent")
	if ok {
		t.Error("expected Lookup to return false for nonexistent model")
	}
}

func TestDiscoverySnapshot_AllModels(t *testing.T) {
	snap := DiscoverySnapshot{
		ByProvider: map[string][]DiscoveredModel{
			"a": {{ID: "m1", Provider: "a"}, {ID: "m2", Provider: "a"}},
			"b": {{ID: "m3", Provider: "b"}},
		},
	}
	all := snap.AllModels()
	if len(all) != 3 {
		t.Errorf("AllModels len = %d, want 3", len(all))
	}
}

func TestModelDiscovery_SnapshotIsCopy(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{ //nolint:errcheck
			"data": []map[string]string{{"id": "model-a"}},
		})
	}))
	defer srv.Close()

	d := NewModelDiscovery([]ProviderEndpoint{{
		Name: "prov", BaseURL: srv.URL, APIKey: "key",
	}}, slog.Default())

	d.refresh()
	snap1 := d.Snapshot()

	// Mutate snap1 — should not affect the discovery's internal state.
	snap1.ByProvider["prov"] = nil

	snap2 := d.Snapshot()
	if len(snap2.ByProvider["prov"]) != 1 {
		t.Error("snapshot mutation affected internal state")
	}
}
