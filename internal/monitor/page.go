package monitor

import (
	_ "embed"
	"net/http"
)

//go:embed page.html
var pageHTML []byte

// HandlePage serves the embedded monitor dashboard HTML.
func HandlePage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(pageHTML) //nolint:errcheck
}
