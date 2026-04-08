#!/usr/bin/env bash
set -euo pipefail

REPO="nogo/wire"
INSTALL_DIR="$HOME/.local/bin"
SERVICE_DIR="$HOME/.config/systemd/user"
BINARY="$INSTALL_DIR/wire"

die() { echo "error: $*" >&2; exit 1; }

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
  x86_64)  ARCH="amd64" ;;
  aarch64) ARCH="arm64" ;;
  *)       die "unsupported architecture: $ARCH" ;;
esac

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
[ "$OS" = "linux" ] || die "unsupported OS: $OS (linux only)"

# Resolve version: use argument, or fetch latest
VERSION="${1:-}"
if [ -z "$VERSION" ]; then
  VERSION=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep '"tag_name"' | head -1 | cut -d'"' -f4)
  [ -n "$VERSION" ] || die "failed to fetch latest release"
fi

# Strip leading v for archive name
VERSION_NUM="${VERSION#v}"

# Check if already installed at this version
if [ -x "$BINARY" ]; then
  CURRENT=$("$BINARY" version 2>/dev/null | awk '{print $2}' || echo "unknown")
  if [ "$CURRENT" = "$VERSION_NUM" ]; then
    echo "wire $VERSION_NUM already installed"
    exit 0
  fi
  echo "upgrading wire $CURRENT -> $VERSION_NUM"
else
  echo "installing wire $VERSION_NUM"
fi

# Download and extract
TARBALL="wire_${VERSION_NUM}_${OS}_${ARCH}.tar.gz"
URL="https://github.com/$REPO/releases/download/${VERSION}/${TARBALL}"
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "downloading $URL"
curl -fsSL -o "$TMPDIR/$TARBALL" "$URL" || die "download failed"
tar -xzf "$TMPDIR/$TARBALL" -C "$TMPDIR"

# Install binary
install -Dm755 "$TMPDIR/wire" "$BINARY"
echo "installed $BINARY"

# Install systemd user service
mkdir -p "$SERVICE_DIR"
cat > "$SERVICE_DIR/wire.service" <<'EOF'
[Unit]
Description=Wire inference gateway
After=network.target

[Service]
Type=simple
ExecStart=%h/.local/bin/wire up
Restart=on-failure
RestartSec=5
Environment=WIRE_CONFIG=%h/.config/wire/config.yaml

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
echo "installed systemd user service"

# Enable but don't start (user may need to configure first)
if ! systemctl --user is-enabled wire &>/dev/null; then
  systemctl --user enable wire
  echo "enabled wire.service"
fi

echo ""
echo "done. commands:"
echo "  systemctl --user start wire     # start now"
echo "  systemctl --user status wire    # check status"
echo "  journalctl --user -u wire -f    # follow logs"
echo "  loginctl enable-linger $USER    # start on boot without login"
