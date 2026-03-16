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

func TestDispatchSwapMissingArgs(t *testing.T) {
	cases := [][]string{
		{"swap"},
		{"swap", "mybackend"},
	}
	for _, args := range cases {
		err := dispatch(args)
		if err == nil {
			t.Errorf("dispatch(%v): expected error, got nil", args)
			continue
		}
		if !strings.Contains(err.Error(), "swap") {
			t.Errorf("dispatch(%v): error should mention swap, got: %v", args, err)
		}
	}
}

func TestListenURL(t *testing.T) {
	cases := []struct {
		listen string
		want   string
	}{
		{":8080", "http://localhost:8080"},
		{"0.0.0.0:9000", "http://0.0.0.0:9000"},
		{"127.0.0.1:8080", "http://127.0.0.1:8080"},
	}
	for _, c := range cases {
		got := listenURL(c.listen)
		if got != c.want {
			t.Errorf("listenURL(%q) = %q, want %q", c.listen, got, c.want)
		}
	}
}
