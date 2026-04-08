package config

import (
	"context"
	"log/slog"
	"path/filepath"
	"time"

	"github.com/fsnotify/fsnotify"
)

const debounceDuration = 500 * time.Millisecond

// Watch monitors the config file for changes and emits validated *Config values
// on the returned channel. The channel is closed when ctx is cancelled.
// Invalid configs are logged and discarded; the running config is never replaced
// with a broken one.
//
// The watcher targets the directory containing the config file so that
// editor write-rename patterns (vim, sed -i) are detected correctly — they
// delete the inode, which breaks file-level watches.
func Watch(ctx context.Context, log *slog.Logger) (<-chan *Config, error) {
	path := Path()
	dir := filepath.Dir(path)
	base := filepath.Base(path)

	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	if err := watcher.Add(dir); err != nil {
		watcher.Close()
		return nil, err
	}

	ch := make(chan *Config, 1)

	go func() {
		defer close(ch)
		defer watcher.Close()

		var timer *time.Timer
		var timerC <-chan time.Time

		for {
			select {
			case <-ctx.Done():
				if timer != nil {
					timer.Stop()
				}
				return

			case event, ok := <-watcher.Events:
				if !ok {
					return
				}
				if filepath.Base(event.Name) != base {
					continue
				}
				if timer != nil {
					timer.Stop()
				}
				timer = time.NewTimer(debounceDuration)
				timerC = timer.C

			case <-timerC:
				timerC = nil
				timer = nil
				cfg, err := Load()
				if err != nil {
					log.Error("config reload failed", "err", err)
					continue
				}
				select {
				case ch <- cfg:
				default:
					log.Warn("config channel full, dropping reload")
				}

			case err, ok := <-watcher.Errors:
				if !ok {
					return
				}
				log.Error("fsnotify error", "err", err)
			}
		}
	}()

	return ch, nil
}
