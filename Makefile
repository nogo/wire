.PHONY: build run test lint clean

BIN := wire
PKG := ./cmd/wire

build:
	go build -o $(BIN) $(PKG)

run: build
	./$(BIN)

test:
	go test ./...

lint:
	golangci-lint run

clean:
	rm -f $(BIN) coverage.out
