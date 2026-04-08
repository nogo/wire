package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
)

// ---- prefixHash tests -------------------------------------------------------

func TestPrefixHash_Stability(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"Hi"}]}`
	h1 := prefixHash([]byte(body))
	h2 := prefixHash([]byte(body))
	if h1 == 0 {
		t.Fatal("expected non-zero hash for valid system message")
	}
	if h1 != h2 {
		t.Errorf("hash not stable: %d != %d", h1, h2)
	}
}

func TestPrefixHash_DifferentSystemPrompts(t *testing.T) {
	body1 := `{"model":"m","messages":[{"role":"system","content":"You are a coder."}]}`
	body2 := `{"model":"m","messages":[{"role":"system","content":"You are a chef."}]}`
	if prefixHash([]byte(body1)) == prefixHash([]byte(body2)) {
		t.Error("different system prompts should produce different hashes")
	}
}

func TestPrefixHash_FallbackToFirstUser(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":"Hello, world!"}]}`
	h := prefixHash([]byte(body))
	if h == 0 {
		t.Error("expected non-zero hash using first user message as fallback")
	}
}

func TestPrefixHash_EmptyMessages(t *testing.T) {
	body := `{"model":"m","messages":[]}`
	if h := prefixHash([]byte(body)); h != 0 {
		t.Errorf("empty messages should return 0, got %d", h)
	}
}

func TestPrefixHash_ParseError(t *testing.T) {
	if h := prefixHash([]byte(`not json`)); h != 0 {
		t.Errorf("parse error should return 0, got %d", h)
	}
}

func TestPrefixHash_NoMessages(t *testing.T) {
	if h := prefixHash([]byte(`{"model":"m"}`)); h != 0 {
		t.Errorf("missing messages should return 0, got %d", h)
	}
}

func TestPrefixHash_Unicode(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"system","content":"你好，世界！🌍"}]}`
	h := prefixHash([]byte(body))
	if h == 0 {
		t.Error("unicode system prompt should produce non-zero hash")
	}
}

// ---- injectSlot tests -------------------------------------------------------

func TestInjectSlot_Basic(t *testing.T) {
	body := []byte(`{"model":"m","messages":[]}`)
	result := injectSlot(body, 3)

	var parsed map[string]any
	if err := json.Unmarshal(result, &parsed); err != nil {
		t.Fatalf("injected body is not valid JSON: %v\nbody: %s", err, result)
	}
	slot, ok := parsed["id_slot"]
	if !ok {
		t.Fatal("id_slot field missing from injected body")
	}
	if int(slot.(float64)) != 3 {
		t.Errorf("id_slot: got %v, want 3", slot)
	}
	if parsed["model"] != "m" {
		t.Errorf("model field should be preserved, got %v", parsed["model"])
	}
}

func TestInjectSlot_SlotZero(t *testing.T) {
	body := []byte(`{"model":"x"}`)
	result := injectSlot(body, 0)
	var parsed map[string]any
	if err := json.Unmarshal(result, &parsed); err != nil {
		t.Fatalf("invalid JSON: %v\nbody: %s", err, result)
	}
	if int(parsed["id_slot"].(float64)) != 0 {
		t.Errorf("id_slot: got %v, want 0", parsed["id_slot"])
	}
}

func TestInjectSlot_WithSpaces(t *testing.T) {
	body := []byte(`{ "model": "m" }`)
	result := injectSlot(body, 1)
	var parsed map[string]any
	if err := json.Unmarshal(result, &parsed); err != nil {
		t.Fatalf("invalid JSON: %v\nbody: %s", err, result)
	}
}

func TestInjectSlot_NoBrace(t *testing.T) {
	body := []byte(`not-an-object`)
	result := injectSlot(body, 0)
	if !bytes.Equal(result, body) {
		t.Error("body without opening brace should be returned unchanged")
	}
}

// ---- affinityTable tests ----------------------------------------------------

func newTestTable(model string, slots int) *affinityTable {
	return newAffinityTable(map[string]int{model: slots})
}

func TestAssign_HashZeroNoAffinity(t *testing.T) {
	at := newTestTable("org/model", 2)
	slotID, ok := at.Assign("org/model", 0)
	if ok || slotID != -1 {
		t.Errorf("hash=0 should return (-1, false), got (%d, %v)", slotID, ok)
	}
}

func TestAssign_UnknownModel(t *testing.T) {
	at := newTestTable("org/model", 2)
	slotID, ok := at.Assign("unknown/model", 42)
	if ok || slotID != -1 {
		t.Errorf("unknown model should return (-1, false), got (%d, %v)", slotID, ok)
	}
}

func TestAssign_SingleSlot_AlwaysSlotZero(t *testing.T) {
	at := newTestTable("org/model", 1)
	for _, hash := range []uint64{1, 2, 3, 999} {
		slotID, ok := at.Assign("org/model", hash)
		if !ok || slotID != 0 {
			t.Errorf("hash=%d: want (0, true), got (%d, %v)", hash, slotID, ok)
		}
	}
}

func TestAssign_SameHashSameSlot(t *testing.T) {
	at := newTestTable("org/model", 4)
	const hash = uint64(0xdeadbeef)
	first, ok := at.Assign("org/model", hash)
	if !ok {
		t.Fatal("first assign should succeed")
	}
	for i := range 5 {
		slotID, ok := at.Assign("org/model", hash)
		if !ok || slotID != first {
			t.Errorf("iteration %d: want (%d, true), got (%d, %v)", i, first, slotID, ok)
		}
	}
}

func TestAssign_DifferentHashesDifferentSlots(t *testing.T) {
	at := newTestTable("org/model", 4)
	hashes := []uint64{1, 2, 3, 4}
	slots := make(map[uint64]int)
	for _, h := range hashes {
		slotID, ok := at.Assign("org/model", h)
		if !ok {
			t.Fatalf("hash=%d: assign failed unexpectedly", h)
		}
		slots[h] = slotID
	}
	// All 4 hashes on a 4-slot model should get distinct slots.
	seen := make(map[int]bool)
	for h, s := range slots {
		if seen[s] {
			t.Errorf("hash %d got duplicate slot %d", h, s)
		}
		seen[s] = true
	}
}

func TestAssign_LRUEviction(t *testing.T) {
	at := newTestTable("org/model", 2) // 2 slots

	// Fill both slots.
	slot1, _ := at.Assign("org/model", 100)
	slot2, _ := at.Assign("org/model", 200)
	if slot1 == slot2 {
		t.Fatal("two different hashes should map to different slots")
	}

	// Access hash 200 to make it recently used (hash 100 becomes oldest).
	at.Assign("org/model", 200) //nolint:errcheck

	// Add a third hash; should evict hash 100 and reuse its slot.
	slot3, ok := at.Assign("org/model", 300)
	if !ok {
		t.Fatal("third assign should succeed via LRU eviction")
	}
	if slot3 != slot1 {
		t.Errorf("evicted slot should be reused: want %d, got %d", slot1, slot3)
	}

	// Hash 200 should still be assigned to its original slot.
	s, ok := at.Assign("org/model", 200)
	if !ok || s != slot2 {
		t.Errorf("hash 200 should still hold slot %d, got (%d, %v)", slot2, s, ok)
	}
}

func TestAssign_ConcurrentSafety(t *testing.T) {
	at := newTestTable("org/model", 4)
	var wg sync.WaitGroup
	for i := range 50 {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			hash := uint64(i%5) + 1 // 5 distinct hashes on 4 slots
			slotID, ok := at.Assign("org/model", hash)
			if !ok || slotID < 0 || slotID >= 4 {
				t.Errorf("goroutine %d: invalid assignment (%d, %v)", i, slotID, ok)
			}
		}(i)
	}
	wg.Wait()
}

// ---- integration test -------------------------------------------------------

func TestIntegration_SameSystemPrompt_SameSlot(t *testing.T) {
	var (
		mu       sync.Mutex
		received [][]byte
	)

	// Backend captures proxied request bodies.
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		received = append(received, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"t","object":"chat.completion","model":"test/model","choices":[]}`)
	}))
	defer backend.Close()

	cfg := testConfig("test/model", 2, backend.URL)
	mgr := &mockManager{
		addrs:   map[string]string{"test/model": backend.URL},
		healthy: map[string]bool{"test/model": true},
	}
	gw := newGateway(cfg, mgr, nil, slog.Default())
	gwServer := httptest.NewServer(gw.server.Handler)
	defer gwServer.Close()

	reqBody := `{"model":"test/model","messages":[{"role":"system","content":"You are a test assistant."},{"role":"user","content":"Hello"}]}`

	sendRequest := func() {
		t.Helper()
		resp, err := http.Post(gwServer.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		resp.Body.Close()
	}

	sendRequest()
	sendRequest()

	mu.Lock()
	defer mu.Unlock()

	if len(received) != 2 {
		t.Fatalf("backend received %d requests, want 2", len(received))
	}

	extractSlot := func(body []byte) (int, bool) {
		var p struct {
			IDSlot *int `json:"id_slot"`
		}
		if err := json.Unmarshal(body, &p); err != nil || p.IDSlot == nil {
			return -1, false
		}
		return *p.IDSlot, true
	}

	slot1, ok1 := extractSlot(received[0])
	slot2, ok2 := extractSlot(received[1])

	if !ok1 || !ok2 {
		t.Fatalf("id_slot missing from proxied body: req1=%q req2=%q", received[0], received[1])
	}
	if slot1 != slot2 {
		t.Errorf("same system prompt should route to same slot: got %d and %d", slot1, slot2)
	}
}

func TestIntegration_Fallback_NoSlotOnHashZero(t *testing.T) {
	var (
		mu       sync.Mutex
		received []byte
	)

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		received = body
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"t","object":"chat.completion","model":"test/model","choices":[]}`)
	}))
	defer backend.Close()

	cfg := testConfig("test/model", 2, backend.URL)
	mgr := &mockManager{
		addrs:   map[string]string{"test/model": backend.URL},
		healthy: map[string]bool{"test/model": true},
	}
	gw := newGateway(cfg, mgr, nil, slog.Default())
	gwServer := httptest.NewServer(gw.server.Handler)
	defer gwServer.Close()

	// Empty messages → hash=0 → no id_slot injected.
	reqBody := `{"model":"test/model","messages":[]}`
	resp, err := http.Post(gwServer.URL+"/v1/chat/completions", "application/json", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	resp.Body.Close()

	mu.Lock()
	defer mu.Unlock()

	var p struct {
		IDSlot *int `json:"id_slot"`
	}
	json.Unmarshal(received, &p) //nolint:errcheck
	if p.IDSlot != nil {
		t.Errorf("empty messages should not inject id_slot, got %d", *p.IDSlot)
	}
}
