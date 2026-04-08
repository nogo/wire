package cli

import (
	"strings"
	"testing"
)

func TestDispatchNoArgs(t *testing.T) {
	if err := dispatch([]string{}); err != nil {
		t.Errorf("expected no error for empty args, got: %v", err)
	}
}

func TestDispatchHelp(t *testing.T) {
	for _, arg := range []string{"help", "--help", "-h"} {
		if err := dispatch([]string{arg}); err != nil {
			t.Errorf("dispatch(%q): unexpected error: %v", arg, err)
		}
	}
}

func TestDispatchUnknownCommand(t *testing.T) {
	err := dispatch([]string{"bogus"})
	if err == nil {
		t.Fatal("expected error for unknown command, got nil")
	}
	if !strings.Contains(err.Error(), "bogus") {
		t.Errorf("error should mention unknown command, got: %v", err)
	}
}

func TestDispatchVersion(t *testing.T) {
	for _, arg := range []string{"version", "--version", "-v"} {
		if err := dispatch([]string{arg}); err != nil {
			t.Errorf("dispatch(%q): unexpected error: %v", arg, err)
		}
	}
}
