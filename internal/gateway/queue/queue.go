package queue

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// Priority determines the time budget for a queued request.
type Priority int

const (
	PriorityHigh   Priority = 0 // chat completions: 30s budget
	PriorityMedium Priority = 1 // completions: 15s budget
	PriorityLow    Priority = 2 // embeddings/rerank: 5s budget
)

var defaultBudgets = [3]time.Duration{
	30 * time.Second,
	15 * time.Second,
	5 * time.Second,
}

var (
	// ErrQueueFull is returned when the global depth cap is exceeded.
	ErrQueueFull = errors.New("queue at capacity")
	// ErrModelEvicted is returned when the model is evicted while the ticket is waiting.
	ErrModelEvicted = errors.New("model evicted")
)

// LoadFunc loads a model and returns its backend address and a release function.
// The release function must be called when the proxied request completes.
type LoadFunc func(ctx context.Context, model string) (addr string, release func(), err error)

// Ticket represents a queued request. Returned immediately by Enqueue.
type Ticket struct {
	ready chan string
	mu    sync.Mutex
	err   error
	rel   func()
	once  sync.Once
}

// Ready returns a channel that receives the backend address on success,
// or is closed without a value on failure. Call Err() to inspect the cause.
func (t *Ticket) Ready() <-chan string {
	return t.ready
}

// Err returns the failure cause. Must be called after Ready() closes or is drained.
func (t *Ticket) Err() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.err
}

// Release decrements the in-flight counter. Must be called exactly once after the request completes.
func (t *Ticket) Release() {
	t.once.Do(func() {
		t.mu.Lock()
		rel := t.rel
		t.mu.Unlock()
		if rel != nil {
			rel()
		}
	})
}

func (t *Ticket) succeed(addr string, rel func()) {
	t.mu.Lock()
	t.rel = rel
	t.mu.Unlock()
	t.ready <- addr
}

func (t *Ticket) fail(err error) {
	t.mu.Lock()
	t.err = err
	t.mu.Unlock()
	close(t.ready)
}

// modelQueue serializes initial loading for a single model.
// Concurrent callers coalesce: the first (primary) calls the loader; subsequent callers
// wait for the primary to finish, then call the loader themselves (which returns immediately
// since the model is already loaded at that point).
type modelQueue struct {
	mu          sync.Mutex
	loading     bool
	loadDone    chan struct{}
	evictCtx    context.Context
	evictCancel context.CancelFunc
}

func (mq *modelQueue) load(ctx context.Context, loader LoadFunc, model string) (string, func(), error) {
	mq.mu.Lock()
	if mq.loading {
		done := mq.loadDone
		mq.mu.Unlock()
		select {
		case <-done:
		case <-ctx.Done():
			return "", nil, ctx.Err()
		}
		// Primary finished. Model is now loaded; this call returns immediately.
		return loader(ctx, model)
	}
	// This goroutine is the primary loader.
	mq.loading = true
	done := make(chan struct{})
	mq.loadDone = done
	mq.mu.Unlock()

	addr, rel, err := loader(ctx, model)

	mq.mu.Lock()
	mq.loading = false
	mq.mu.Unlock()
	close(done)

	return addr, rel, err
}

// Queue manages per-model request queuing with time-bounded priority budgets.
type Queue struct {
	mu       sync.Mutex
	models   map[string]*modelQueue
	depth    atomic.Int64
	maxDepth int64
	loader   LoadFunc
	budgets  [3]time.Duration
}

// New creates a Queue with the given depth cap and loader function.
func New(maxDepth int, loader LoadFunc) *Queue {
	return &Queue{
		models:   make(map[string]*modelQueue),
		maxDepth: int64(maxDepth),
		loader:   loader,
		budgets:  defaultBudgets,
	}
}

// Enqueue registers a request for the given model at the specified priority.
// Returns a Ticket immediately; the caller waits on ticket.Ready() for the backend address.
func (q *Queue) Enqueue(model string, prio Priority, reqCtx context.Context) *Ticket {
	if q.depth.Add(1) > q.maxDepth {
		q.depth.Add(-1)
		t := &Ticket{ready: make(chan string, 1)}
		t.fail(ErrQueueFull)
		return t
	}

	q.mu.Lock()
	mq, ok := q.models[model]
	if !ok {
		evictCtx, evictCancel := context.WithCancel(context.Background())
		mq = &modelQueue{
			evictCtx:    evictCtx,
			evictCancel: evictCancel,
		}
		q.models[model] = mq
	}
	q.mu.Unlock()

	t := &Ticket{ready: make(chan string, 1)}

	go func() {
		defer q.depth.Add(-1)

		// Combined context: cancelled by request cancellation or the priority budget.
		ctx, cancel := context.WithTimeout(reqCtx, q.budgets[prio])
		defer cancel()

		// Also watch for model eviction; cancel the load context if evicted.
		go func() {
			select {
			case <-mq.evictCtx.Done():
				cancel()
			case <-ctx.Done():
			}
		}()

		addr, rel, err := mq.load(ctx, q.loader, model)
		if err != nil {
			select {
			case <-mq.evictCtx.Done():
				t.fail(ErrModelEvicted)
			default:
				t.fail(err)
			}
			return
		}

		// Load succeeded, but eviction may have raced. Check before serving.
		select {
		case <-mq.evictCtx.Done():
			if rel != nil {
				rel()
			}
			t.fail(ErrModelEvicted)
			return
		default:
		}

		t.succeed(addr, rel)
	}()

	return t
}

// CancelModel cancels all tickets for the given model with ErrModelEvicted.
// Call this when a model is being evicted by the process manager.
func (q *Queue) CancelModel(model string) {
	q.mu.Lock()
	mq, ok := q.models[model]
	if ok {
		delete(q.models, model)
	}
	q.mu.Unlock()
	if ok {
		mq.evictCancel()
	}
}
