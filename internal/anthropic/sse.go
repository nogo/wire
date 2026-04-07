package anthropic

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// sseEvent holds one parsed Anthropic SSE event.
type sseEvent struct {
	Type string
	Data string
}

// parseSSE reads Anthropic SSE events from r until EOF or error.
// Each complete event (with non-empty data) is sent to events.
// The channel is closed when done.
func parseSSE(r io.Reader, events chan<- sseEvent) {
	defer close(events)

	scanner := bufio.NewScanner(r)
	var curType, curData string

	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event:"):
			curType = strings.TrimSpace(line[6:])
		case strings.HasPrefix(line, "data:"):
			data := strings.TrimSpace(line[5:])
			if data == "[DONE]" {
				continue
			}
			curData = data
		case line == "":
			if curData != "" {
				events <- sseEvent{Type: curType, Data: curData}
			}
			curType = ""
			curData = ""
		}
	}

	// Send any trailing event not followed by a blank line.
	if curData != "" {
		events <- sseEvent{Type: curType, Data: curData}
	}
}

// --- OpenAI SSE chunk types ---

type oaiChunk struct {
	ID      string           `json:"id"`
	Object  string           `json:"object"`
	Created int64            `json:"created"`
	Model   string           `json:"model"`
	Choices []oaiChunkChoice `json:"choices"`
}

type oaiChunkChoice struct {
	Index        int      `json:"index"`
	Delta        oaiDelta `json:"delta"`
	FinishReason *string  `json:"finish_reason"`
}

type oaiDelta struct {
	Role      string             `json:"role,omitempty"`
	Content   *string            `json:"content,omitempty"`
	ToolCalls []oaiToolCallChunk `json:"tool_calls,omitempty"`
}

type oaiToolCallChunk struct {
	Index    int          `json:"index"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function oaiFuncChunk `json:"function"`
}

type oaiFuncChunk struct {
	Name      string  `json:"name,omitempty"`
	Arguments *string `json:"arguments,omitempty"`
}

// TranslateStream reads an Anthropic SSE response body and writes equivalent
// OpenAI SSE chunks to w. model is reported as-is in each chunk.
// The caller must set appropriate response headers before calling.
func TranslateStream(anthropicBody io.Reader, w http.ResponseWriter, model string) {
	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	created := time.Now().Unix()

	type blockState struct {
		oaiIndex  int // index in the OpenAI tool_calls array
		isToolUse bool
		id        string
		name      string
	}

	blocks := map[int]*blockState{}
	toolCallIndex := 0

	emit := func(chunk oaiChunk) {
		b, err := json.Marshal(chunk)
		if err != nil {
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", b)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
	}

	sseEvents := make(chan sseEvent, 64)
	go parseSSE(anthropicBody, sseEvents)

	for ev := range sseEvents {
		switch ev.Type {
		case "message_start":
			empty := ""
			emit(oaiChunk{
				ID:      id,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []oaiChunkChoice{{
					Index:        0,
					Delta:        oaiDelta{Role: "assistant", Content: &empty},
					FinishReason: nil,
				}},
			})

		case "content_block_start":
			var cbs struct {
				Index        int `json:"index"`
				ContentBlock struct {
					Type string `json:"type"`
					ID   string `json:"id"`
					Name string `json:"name"`
				} `json:"content_block"`
			}
			if err := json.Unmarshal([]byte(ev.Data), &cbs); err != nil {
				slog.Warn("anthropic: malformed content_block_start", "error", err)
				continue
			}
			bs := &blockState{isToolUse: cbs.ContentBlock.Type == "tool_use"}
			blocks[cbs.Index] = bs
			if cbs.ContentBlock.Type == "tool_use" {
				bs.id = cbs.ContentBlock.ID
				bs.name = cbs.ContentBlock.Name
				bs.oaiIndex = toolCallIndex
				toolCallIndex++
				emptyArgs := ""
				emit(oaiChunk{
					ID:      id,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []oaiChunkChoice{{
						Index: 0,
						Delta: oaiDelta{
							ToolCalls: []oaiToolCallChunk{{
								Index: bs.oaiIndex,
								ID:    bs.id,
								Type:  "function",
								Function: oaiFuncChunk{
									Name:      bs.name,
									Arguments: &emptyArgs,
								},
							}},
						},
						FinishReason: nil,
					}},
				})
			}

		case "content_block_delta":
			var cbd struct {
				Index int `json:"index"`
				Delta struct {
					Type        string `json:"type"`
					Text        string `json:"text"`
					PartialJSON string `json:"partial_json"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(ev.Data), &cbd); err != nil {
				slog.Warn("anthropic: malformed content_block_delta", "error", err)
				continue
			}
			bs := blocks[cbd.Index]
			switch cbd.Delta.Type {
			case "text_delta":
				text := cbd.Delta.Text
				emit(oaiChunk{
					ID:      id,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   model,
					Choices: []oaiChunkChoice{{
						Index:        0,
						Delta:        oaiDelta{Content: &text},
						FinishReason: nil,
					}},
				})
			case "input_json_delta":
				if bs != nil && bs.isToolUse {
					args := cbd.Delta.PartialJSON
					emit(oaiChunk{
						ID:      id,
						Object:  "chat.completion.chunk",
						Created: created,
						Model:   model,
						Choices: []oaiChunkChoice{{
							Index: 0,
							Delta: oaiDelta{
								ToolCalls: []oaiToolCallChunk{{
									Index: bs.oaiIndex,
									Function: oaiFuncChunk{
										Arguments: &args,
									},
								}},
							},
							FinishReason: nil,
						}},
					})
				}
			}

		case "message_delta":
			var md struct {
				Delta struct {
					StopReason string `json:"stop_reason"`
				} `json:"delta"`
			}
			if err := json.Unmarshal([]byte(ev.Data), &md); err != nil {
				slog.Warn("anthropic: malformed message_delta", "error", err)
				continue
			}
			reason := mapStopReason(md.Delta.StopReason)
			emit(oaiChunk{
				ID:      id,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []oaiChunkChoice{{
					Index:        0,
					Delta:        oaiDelta{},
					FinishReason: &reason,
				}},
			})

		case "message_stop":
			fmt.Fprintf(w, "data: [DONE]\n\n")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}

		default:
			// ping and other unknown events are silently ignored.
		}
	}
}
