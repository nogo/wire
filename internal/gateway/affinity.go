package gateway

import (
	"encoding/json"
	"hash/fnv"
	"strconv"
	"sync"

	"github.com/nogo/wire/internal/config"
)

// affinityTable maps prefix hashes to slot IDs per backend, enabling KV cache
// reuse in llama-server by routing requests with matching prefixes to the same slot.
type affinityTable struct {
	mu       sync.Mutex
	backends map[string]*backendSlots
}

type backendSlots struct {
	slotCount   int
	assignments map[uint64]int // prefix hash → slot ID
	lru         []uint64       // eviction order, oldest first
}

func newAffinityTable(backends map[string]config.Backend) *affinityTable {
	at := &affinityTable{backends: make(map[string]*backendSlots, len(backends))}
	for name, b := range backends {
		slots := b.Slots
		if slots <= 0 {
			slots = 1
		}
		at.backends[name] = &backendSlots{
			slotCount:   slots,
			assignments: make(map[uint64]int),
		}
	}
	return at
}

// Assign returns the slot ID to use for a request with the given prefix hash on
// the named backend. Returns (-1, false) when hash is 0 or the backend is unknown.
func (at *affinityTable) Assign(backend string, hash uint64) (int, bool) {
	if hash == 0 {
		return -1, false
	}

	at.mu.Lock()
	defer at.mu.Unlock()

	bs, ok := at.backends[backend]
	if !ok {
		return -1, false
	}

	// Single-slot backends always use slot 0; no table needed.
	if bs.slotCount == 1 {
		return 0, true
	}

	// Existing assignment: refresh LRU position.
	if slotID, exists := bs.assignments[hash]; exists {
		for i, h := range bs.lru {
			if h == hash {
				bs.lru = append(bs.lru[:i], bs.lru[i+1:]...)
				break
			}
		}
		bs.lru = append(bs.lru, hash)
		return slotID, true
	}

	// Free slot available: assign the next unused slot ID.
	if len(bs.assignments) < bs.slotCount {
		used := make(map[int]bool, len(bs.assignments))
		for _, s := range bs.assignments {
			used[s] = true
		}
		for i := range bs.slotCount {
			if !used[i] {
				bs.assignments[hash] = i
				bs.lru = append(bs.lru, hash)
				return i, true
			}
		}
	}

	// All slots assigned: evict the oldest LRU entry and reuse its slot.
	oldest := bs.lru[0]
	slotID := bs.assignments[oldest]
	bs.lru = bs.lru[1:]
	delete(bs.assignments, oldest)
	bs.assignments[hash] = slotID
	bs.lru = append(bs.lru, hash)
	return slotID, true
}

// prefixHash returns a 64-bit FNV-1a hash of the dominant stable prefix in the
// request: the system message content, falling back to the first user message.
// Returns 0 on parse error or when no usable prefix is found.
func prefixHash(body []byte) uint64 {
	var payload struct {
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return 0
	}

	var text string
	for _, m := range payload.Messages {
		if m.Role == "system" {
			text = m.Content
			break
		}
	}
	if text == "" {
		for _, m := range payload.Messages {
			if m.Role == "user" {
				text = m.Content
				break
			}
		}
	}
	if text == "" {
		return 0
	}

	h := fnv.New64a()
	h.Write([]byte(text)) //nolint:errcheck
	return h.Sum64()
}

// injectSlot inserts `"id_slot":N,` immediately after the opening brace of a
// JSON object body. The body must be a valid JSON object (guaranteed by the
// model extraction that precedes this call).
func injectSlot(body []byte, slotID int) []byte {
	for i, b := range body {
		if b == '{' {
			field := `"id_slot":` + strconv.Itoa(slotID) + `,`
			result := make([]byte, 0, len(body)+len(field))
			result = append(result, body[:i+1]...)
			result = append(result, field...)
			result = append(result, body[i+1:]...)
			return result
		}
	}
	return body
}
