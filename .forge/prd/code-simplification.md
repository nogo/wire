# Code Simplification

## Problem Statement

Several areas of the codebase carry complexity that does not earn its keep. Four specific patterns were identified during an architecture review:

1. `process.LoadTicket` and `queue.Ticket` are structurally identical types — same fields, same methods, same internal helpers — defined separately in two packages. Any change to the ticket contract must be made twice.

2. `queue.modelQueue` serialises concurrent loads for a single model via a `loading` flag and a `loadDone` channel. The process manager's `procState.ready` channel already does this — concurrent `LoadOrGet` calls for a loading model all block on the same `ready` channel. The Queue layer re-implements coalescing that the Manager already owns.

3. `hub.go` contains a hand-rolled `itoa` function to convert an integer to a decimal string, explicitly to avoid importing `strconv`. The function is longer than the import it avoids and is a maintenance hazard.

4. HF cache scanning is performed independently by two packages — the gateway and the VRAM tracker — against the same filesystem tree, with different TTLs and different output shapes. This duplication is addressed structurally by the model-discovery PRD, but the underlying cause (no shared source of truth for discovered models) is worth naming as a simplification target in its own right.

## Solution

Consolidate the two ticket types into one, defined in the `process` package (the origin of the ticket concept), and have the queue package use it directly. Remove `modelQueue`'s coalescing layer and let the process manager's existing state machine own that responsibility. Replace `itoa` with `strconv.Itoa`. The HF cache duplication is resolved as a consequence of the model-discovery PRD and is tracked there.

## User Stories

1. As a developer modifying the ticket contract (Ready, Err, Release), I want a single type to update so that I cannot introduce a divergence between the process and queue packages.

2. As a developer reading `queue.Enqueue`, I want the coalescing logic removed so that the function's responsibility is limited to depth-capping, priority budgeting, and delegating to the loader — not reimplementing load serialisation.

3. As a developer reading `hub.go`, I want standard library functions used for standard operations so that I do not have to reason about whether a custom helper has edge cases.

4. As a developer writing tests for the queue, I want the queue's behaviour to be testable without also testing the coalescing logic that belongs to the process manager, so that tests are scoped to what the queue actually owns.

## Implementation Decisions

### Ticket consolidation

`queue.Ticket` is deleted. `queue.Enqueue` returns `*process.LoadTicket`. The `queue` package imports `process` for the type only — no circular dependency since `process` does not import `queue`. All call sites in the gateway that currently receive `*queue.Ticket` are updated to `*process.LoadTicket`; the interface is identical so no behavioural changes are needed.

### modelQueue coalescing removal

`modelQueue.loading`, `modelQueue.loadDone`, and the wait branch in `modelQueue.load` are deleted. `modelQueue.load` becomes a direct call to the loader function. The eviction context (`evictCtx`, `evictCancel`) remains — it is the queue's own concern and unrelated to coalescing.

The process manager's `procState.ready` channel already handles the concurrent-load case correctly: if state is `stateLoading`, subsequent `LoadOrGet` calls read the ready channel and block until loading completes. Removing the Queue's redundant layer does not change observable behaviour.

### itoa removal

`hub.go`'s `itoa` function is deleted and replaced with `strconv.Itoa` at the single call site. `strconv` is added to the import block.

### HF cache duplication

Tracked in the model-discovery PRD. Not duplicated here.

## Testing Decisions

- **Ticket consolidation**: existing tests in `gateway/gateway_test.go` and `gateway/queue/queue_test.go` that construct or assert on tickets continue to pass without modification — only the type name changes at the queue boundary.
- **modelQueue coalescing removal**: `queue_test.go` currently tests the coalescing behaviour (multiple concurrent callers coalescing onto one load). Those tests are deleted — coalescing is now tested in `process/manager_test.go` where it always belonged. The queue tests retain coverage for depth-cap rejection, priority budget expiry, and eviction cancellation.
- **itoa removal**: no dedicated test needed; the function is replaced with a standard library call.
- **Prior art**: `process/manager_test.go` already tests concurrent `LoadOrGet` coalescing with `sync.WaitGroup` and fake command builders. That test suite is the correct home for coalescing coverage.

## Out of Scope

- Redesigning the ticket API (Ready/Err/Release contract is unchanged).
- Changing the queue's priority tiers or budget values.
- Any changes to the process manager's state machine beyond removing the now-redundant wait path that was mirroring the Manager's own behaviour.
- The `HubMessage` tagged union — identified as a smell but not simplified here; fixing it in Go without generics produces more boilerplate than it removes.

## Further Notes

- The ticket consolidation creates a one-way import: `queue` imports `process`. This is already the case in practice (the queue's `LoadFunc` signature mirrors the Manager's `LoadOrGet` return type). Making it explicit via the shared type is more honest than maintaining two identical definitions to avoid the import.
- The coalescing removal makes `modelQueue` a thin eviction-context carrier. If that context is the only remaining reason for `modelQueue` to exist, a follow-up simplification could inline it directly into `Queue` — but that is not in scope here.
