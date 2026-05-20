package queue

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/nogo/wire/internal/process"
)

// testBudgets replaces the production time budgets with fast values suitable for unit tests.
var testBudgets = [3]time.Duration{
	300 * time.Millisecond, // PriorityHigh
	150 * time.Millisecond, // PriorityMedium
	50 * time.Millisecond,  // PriorityLow
}

func newTestQueue(maxDepth int, loader LoadFunc) *Queue {
	q := New(maxDepth, loader)
	q.budgets = testBudgets
	return q
}

func immediateLoader(_ context.Context, _ string) (string, func(), error) {
	return "http://localhost:8080", func() {}, nil
}

// awaitTicket waits for the ticket's Ready channel and returns success/failure.
func awaitTicket(t *testing.T, ticket *process.LoadTicket, timeout time.Duration) (addr string, ok bool) {
	t.Helper()
	select {
	case addr, ok = <-ticket.Ready():
		return addr, ok
	case <-time.After(timeout):
		t.Fatal("timed out waiting for ticket")
		return "", false
	}
}

// ---- Depth cap ----

func TestDepthCap(t *testing.T) {
	// Block the loader so all slots fill up.
	block := make(chan struct{})
	loader := func(ctx context.Context, _ string) (string, func(), error) {
		select {
		case <-block:
		case <-ctx.Done():
		}
		return "http://localhost:8080", func() {}, nil
	}

	q := newTestQueue(3, loader)

	// Fill queue to capacity.
	t1 := q.Enqueue("m", PriorityHigh, context.Background())
	t2 := q.Enqueue("m", PriorityHigh, context.Background())
	t3 := q.Enqueue("m", PriorityHigh, context.Background())

	// Give goroutines time to start.
	time.Sleep(20 * time.Millisecond)

	// Fourth request must be rejected immediately.
	t4 := q.Enqueue("m", PriorityHigh, context.Background())
	select {
	case _, ok := <-t4.Ready():
		if ok {
			t.Fatal("expected channel to be closed (failure), got a value")
		}
		if !errors.Is(t4.Err(), ErrQueueFull) {
			t.Errorf("expected ErrQueueFull, got %v", t4.Err())
		}
	case <-time.After(100 * time.Millisecond):
		t.Fatal("over-capacity ticket did not fail immediately")
	}

	// Unblock and clean up.
	close(block)
	awaitTicket(t, t1, time.Second)
	awaitTicket(t, t2, time.Second)
	awaitTicket(t, t3, time.Second)
}

// ---- Priority time budgets ----

func TestPriorityBudgets(t *testing.T) {
	// Loader blocks longer than low-prio budget but shorter than high-prio budget.
	// With testBudgets: low=50ms, high=300ms. Load delay: 120ms.
	const loadDelay = 120 * time.Millisecond

	loader := func(ctx context.Context, _ string) (string, func(), error) {
		select {
		case <-time.After(loadDelay):
			return "http://localhost:8080", func() {}, nil
		case <-ctx.Done():
			return "", nil, ctx.Err()
		}
	}

	q := newTestQueue(200, loader)

	tHigh := q.Enqueue("m", PriorityHigh, context.Background())
	tLow := q.Enqueue("m", PriorityLow, context.Background())

	// Low-prio (50ms budget) should time out before the 120ms load completes.
	select {
	case _, ok := <-tLow.Ready():
		if ok {
			t.Error("low-prio ticket should have failed (budget exceeded)")
		} else if tLow.Err() == nil {
			t.Error("expected non-nil error on low-prio ticket")
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("low-prio ticket did not fail within its budget")
	}

	// High-prio (300ms budget) should succeed after the 120ms load.
	addr, ok := awaitTicket(t, tHigh, time.Second)
	if !ok {
		t.Fatalf("high-prio ticket failed: %v", tHigh.Err())
	}
	if addr == "" {
		t.Error("expected non-empty addr from high-prio ticket")
	}
}

// ---- Model eviction ----

func TestEviction(t *testing.T) {
	block := make(chan struct{})
	loader := func(ctx context.Context, _ string) (string, func(), error) {
		select {
		case <-block:
			return "http://localhost:8080", func() {}, nil
		case <-ctx.Done():
			return "", nil, ctx.Err()
		}
	}

	q := newTestQueue(200, loader)

	t1 := q.Enqueue("model-b", PriorityHigh, context.Background())
	t2 := q.Enqueue("model-b", PriorityHigh, context.Background())

	// Give goroutines time to enter the load wait.
	time.Sleep(20 * time.Millisecond)

	// Evict model while tickets are queued.
	q.CancelModel("model-b")

	for i, tk := range []*process.LoadTicket{t1, t2} {
		select {
		case _, ok := <-tk.Ready():
			if ok {
				t.Errorf("ticket[%d] should have failed on eviction", i)
			} else if !errors.Is(tk.Err(), ErrModelEvicted) {
				t.Errorf("ticket[%d]: expected ErrModelEvicted, got %v", i, tk.Err())
			}
		case <-time.After(500 * time.Millisecond):
			t.Fatalf("ticket[%d] did not fail after eviction", i)
		}
	}

	close(block) // unblock loader goroutines (they'll see ctx cancelled)
}

// ---- Request context cancellation ----

func TestRequestContextCancellation(t *testing.T) {
	block := make(chan struct{})
	defer close(block)

	loader := func(ctx context.Context, _ string) (string, func(), error) {
		select {
		case <-block:
			return "http://localhost:8080", func() {}, nil
		case <-ctx.Done():
			return "", nil, ctx.Err()
		}
	}

	q := newTestQueue(200, loader)

	reqCtx, reqCancel := context.WithCancel(context.Background())
	tk := q.Enqueue("model-c", PriorityHigh, reqCtx)

	time.Sleep(20 * time.Millisecond)
	reqCancel()

	select {
	case _, ok := <-tk.Ready():
		if ok {
			t.Error("ticket should have failed when request context was cancelled")
		}
		if tk.Err() == nil {
			t.Error("expected non-nil error")
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("ticket did not fail after request context cancellation")
	}
}

// ---- Ticket Release ----

func TestTicketRelease(t *testing.T) {
	released := atomic.Bool{}
	loader := func(_ context.Context, _ string) (string, func(), error) {
		rel := func() { released.Store(true) }
		return "http://localhost:8080", rel, nil
	}

	q := newTestQueue(200, loader)
	tk := q.Enqueue("model-d", PriorityHigh, context.Background())

	addr, ok := awaitTicket(t, tk, time.Second)
	if !ok || addr == "" {
		t.Fatalf("ticket failed: %v", tk.Err())
	}

	if released.Load() {
		t.Error("release should not have been called before Release()")
	}
	tk.Release()
	if !released.Load() {
		t.Error("release should have been called after Release()")
	}

	// Idempotent: second call must not panic or double-release.
	tk.Release()
}

// ---- Priority ordering (budget-based) ----

func TestPriorityOrdering(t *testing.T) {
	// Load takes 80ms on the first successful call; subsequent calls return immediately,
	// matching production behavior where the model is already loaded.
	// With testBudgets: low=50ms (times out), medium=150ms (succeeds), high=300ms (succeeds).
	const loadDelay = 80 * time.Millisecond

	var loaded atomic.Bool
	loader := func(ctx context.Context, _ string) (string, func(), error) {
		if loaded.Load() {
			return "http://localhost:8080", func() {}, nil
		}
		select {
		case <-time.After(loadDelay):
			loaded.Store(true)
			return "http://localhost:8080", func() {}, nil
		case <-ctx.Done():
			return "", nil, ctx.Err()
		}
	}

	q := newTestQueue(200, loader)

	tHigh := q.Enqueue("ordering-model", PriorityHigh, context.Background())
	tMed := q.Enqueue("ordering-model", PriorityMedium, context.Background())
	tLow := q.Enqueue("ordering-model", PriorityLow, context.Background())

	// Low-prio times out first.
	select {
	case _, ok := <-tLow.Ready():
		if ok {
			t.Error("low-prio ticket should have timed out")
		}
	case <-time.After(200 * time.Millisecond):
		t.Fatal("low-prio ticket did not time out within expected window")
	}

	// Medium and high should succeed.
	_, okMed := awaitTicket(t, tMed, time.Second)
	if !okMed {
		t.Errorf("medium-prio ticket failed: %v", tMed.Err())
	}
	_, okHigh := awaitTicket(t, tHigh, time.Second)
	if !okHigh {
		t.Errorf("high-prio ticket failed: %v", tHigh.Err())
	}
}

// ---- Multiple models are independent ----

func TestMultipleModelsIndependent(t *testing.T) {
	q := newTestQueue(200, immediateLoader)

	tA := q.Enqueue("model-x", PriorityHigh, context.Background())
	tB := q.Enqueue("model-y", PriorityHigh, context.Background())

	addrA, okA := awaitTicket(t, tA, time.Second)
	addrB, okB := awaitTicket(t, tB, time.Second)

	if !okA || addrA == "" {
		t.Errorf("model-x ticket failed: %v", tA.Err())
	}
	if !okB || addrB == "" {
		t.Errorf("model-y ticket failed: %v", tB.Err())
	}
}

// ---- Depth counter resets after tickets resolve ----

func TestDepthCounterResets(t *testing.T) {
	q := newTestQueue(2, immediateLoader)

	t1 := q.Enqueue("m", PriorityHigh, context.Background())
	t2 := q.Enqueue("m", PriorityHigh, context.Background())

	awaitTicket(t, t1, time.Second)
	awaitTicket(t, t2, time.Second)

	// Give depth.Add(-1) goroutines time to run.
	time.Sleep(20 * time.Millisecond)

	// Should be able to enqueue again.
	t3 := q.Enqueue("m", PriorityHigh, context.Background())
	_, ok := awaitTicket(t, t3, time.Second)
	if !ok {
		t.Errorf("expected ticket to succeed after depth reset: %v", t3.Err())
	}
}

// ---- ErrQueueFull is not ErrModelEvicted ----

func TestErrorDistinction(t *testing.T) {
	if errors.Is(ErrQueueFull, ErrModelEvicted) {
		t.Error("ErrQueueFull should not be ErrModelEvicted")
	}
	if errors.Is(ErrModelEvicted, ErrQueueFull) {
		t.Error("ErrModelEvicted should not be ErrQueueFull")
	}
}
