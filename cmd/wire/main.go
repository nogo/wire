package main

import (
	"os"

	"github.com/nogo/wire/internal/cli"
)

func main() {
	os.Exit(cli.Run(os.Args[1:]))
}
