package process

import (
	"errors"
	"sync"
)

// ErrModelDraining is returned when a LoadOrGet call finds the target model
// in the stateUnloading state (SIGTERM sent, waiting for exit).
var ErrModelDraining = errors.New("model draining")

// LoadTicket is an asynchronous handle for a model load request returned by
// LoadOrGet. Call Ready() to wait for the backend address.
type LoadTicket struct {
	ready chan string
	mu    sync.Mutex
	err   error
	rel   func()
	once  sync.Once
}

// Ready returns a channel that receives the backend address on success.
// If loading fails the channel is closed without a value; call Err() to
// inspect the cause.
func (t *LoadTicket) Ready() <-chan string { return t.ready }

// Err returns the failure cause after Ready() is drained or closed.
func (t *LoadTicket) Err() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.err
}

// Release decrements the in-flight counter. Must be called exactly once
// after the proxied request completes.
func (t *LoadTicket) Release() {
	t.once.Do(func() {
		t.mu.Lock()
		rel := t.rel
		t.mu.Unlock()
		if rel != nil {
			rel()
		}
	})
}

func (t *LoadTicket) succeed(addr string, rel func()) {
	t.mu.Lock()
	t.rel = rel
	t.mu.Unlock()
	t.ready <- addr
}

func (t *LoadTicket) fail(err error) {
	t.mu.Lock()
	t.err = err
	t.mu.Unlock()
	close(t.ready)
}

func newTicket() *LoadTicket {
	return &LoadTicket{ready: make(chan string, 1)}
}

// NewFailedTicket returns a ticket that is already resolved with the given error.
// Used by test doubles implementing processManager.
func NewFailedTicket(err error) *LoadTicket {
	t := newTicket()
	t.fail(err)
	return t
}

// NewReadyTicket returns a ticket that is already resolved with the given address
// and release function. Used by test doubles implementing processManager.
func NewReadyTicket(addr string, rel func()) *LoadTicket {
	t := newTicket()
	t.succeed(addr, rel)
	return t
}
