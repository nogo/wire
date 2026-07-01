package queue

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"github.com/nogo/wire/internal/process"
)

var (
	// ErrQueueFull is returned when the global depth cap is exceeded.
	ErrQueueFull = errors.New("queue at capacity")
	// ErrModelEvicted is returned when the model is evicted while the ticket is waiting.
	ErrModelEvicted = errors.New("model evicted")
)

// LoadFunc loads a model and returns its backend address and a release function.
// The release function must be called when the proxied request completes.
type LoadFunc func(ctx context.Context, model string) (addr string, release func(), err error)

// modelQueue carries the eviction context for a single model.
type modelQueue struct {
	evictCtx    context.Context
	evictCancel context.CancelFunc
}

func (mq *modelQueue) load(ctx context.Context, loader LoadFunc, model string) (string, func(), error) {
	return loader(ctx, model)
}

// Queue manages per-model request queuing with a caller-supplied load budget.
type Queue struct {
	mu       sync.Mutex
	models   map[string]*modelQueue
	depth    atomic.Int64
	maxDepth int64
	loader   LoadFunc
}

// New creates a Queue with the given depth cap and loader function.
func New(maxDepth int, loader LoadFunc) *Queue {
	return &Queue{
		models:   make(map[string]*modelQueue),
		maxDepth: int64(maxDepth),
		loader:   loader,
	}
}

// Enqueue registers a request for the given model, bounding the wait for the
// model to become ready by budget (see process.LoadBudget — automatic from model
// size). Returns a ticket immediately; the caller waits on ticket.Ready().
func (q *Queue) Enqueue(model string, budget time.Duration, reqCtx context.Context) *process.LoadTicket {
	if q.depth.Add(1) > q.maxDepth {
		q.depth.Add(-1)
		return process.NewFailedTicket(ErrQueueFull)
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

	t := process.NewTicket()

	go func() {
		defer q.depth.Add(-1)

		// Combined context: cancelled by request cancellation or the priority budget.
		ctx, cancel := context.WithTimeout(reqCtx, budget)
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
				t.Fail(ErrModelEvicted)
			default:
				t.Fail(err)
			}
			return
		}

		// Load succeeded, but eviction may have raced. Check before serving.
		select {
		case <-mq.evictCtx.Done():
			if rel != nil {
				rel()
			}
			t.Fail(ErrModelEvicted)
			return
		default:
		}

		t.Succeed(addr, rel)
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
