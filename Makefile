BINARY := wire

.PHONY: build install uninstall test

build:
	go build -o $(BINARY) ./cmd/wire

install: build
	install -Dm755 $(BINARY) $(HOME)/.local/bin/$(BINARY)
	install -Dm644 deploy/wire.service $(HOME)/.config/systemd/user/wire.service
	systemctl --user daemon-reload
	@echo ""
	@echo "Installed wire to $(HOME)/.local/bin/wire"
	@echo "Installed systemd service to $(HOME)/.config/systemd/user/wire.service"
	@echo ""
	@echo "To enable wire on boot (without requiring a login session):"
	@echo "  loginctl enable-linger $(USER)"
	@echo "  systemctl --user enable wire"
	@echo ""
	@echo "To start now:"
	@echo "  systemctl --user start wire"

uninstall:
	systemctl --user stop wire || true
	systemctl --user disable wire || true
	rm -f $(HOME)/.local/bin/$(BINARY)
	rm -f $(HOME)/.config/systemd/user/wire.service
	systemctl --user daemon-reload

test:
	go test ./...
