// Package anthropic translates between OpenAI chat completion format and the
// Anthropic Messages API format. Callers always speak OpenAI to wire; this
// package converts on the fly when the backend has provider: "anthropic".
package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

// --- OpenAI wire types (request) ---

type oaiRequest struct {
	Model       string    `json:"model"`
	Messages    []oaiMsg  `json:"messages"`
	Tools       []oaiTool `json:"tools,omitempty"`
	MaxTokens   int       `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	Stream      bool      `json:"stream,omitempty"`
}

type oaiMsg struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	ToolCalls  []oaiToolCall   `json:"tool_calls,omitempty"`
}

type oaiToolCall struct {
	ID       string      `json:"id"`
	Type     string      `json:"type"`
	Function oaiFunction `json:"function"`
}

type oaiFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type oaiTool struct {
	Type     string     `json:"type"`
	Function oaiFuncDef `json:"function"`
}

type oaiFuncDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// --- Anthropic wire types (request) ---

type antRequest struct {
	Model       string    `json:"model"`
	MaxTokens   int       `json:"max_tokens"`
	System      string    `json:"system,omitempty"`
	Messages    []antMsg  `json:"messages"`
	Tools       []antTool `json:"tools,omitempty"`
	Stream      bool      `json:"stream"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
}

type antMsg struct {
	Role    string     `json:"role"`
	Content []antBlock `json:"content"`
}

type antBlock struct {
	Type string `json:"type"`
	// text block
	Text string `json:"text,omitempty"`
	// tool_use block
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
	// tool_result block
	ToolUseID string `json:"tool_use_id,omitempty"`
	Content   string `json:"content,omitempty"`
}

type antTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

// --- Anthropic wire types (response) ---

type antResponse struct {
	ID         string     `json:"id"`
	Type       string     `json:"type"`
	Role       string     `json:"role"`
	Content    []antBlock `json:"content"`
	Model      string     `json:"model"`
	StopReason string     `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// --- OpenAI wire types (response) ---

type oaiResponse struct {
	ID      string      `json:"id"`
	Object  string      `json:"object"`
	Created int64       `json:"created"`
	Model   string      `json:"model"`
	Choices []oaiChoice `json:"choices"`
	Usage   oaiUsage    `json:"usage"`
}

type oaiChoice struct {
	Index        int       `json:"index"`
	Message      oaiMsgOut `json:"message"`
	FinishReason string    `json:"finish_reason"`
}

type oaiMsgOut struct {
	Role      string        `json:"role"`
	Content   *string       `json:"content"` // null when only tool_calls
	ToolCalls []oaiToolCall `json:"tool_calls,omitempty"`
}

type oaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// TranslateRequest converts an OpenAI chat completion request to Anthropic
// Messages API format. model is the Anthropic model name from the backend
// config and overrides whatever is in the incoming request.
func TranslateRequest(openaiReq []byte, model string) ([]byte, error) {
	var req oaiRequest
	if err := json.Unmarshal(openaiReq, &req); err != nil {
		return nil, fmt.Errorf("parse openai request: %w", err)
	}

	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = 8192
	}

	// Extract system message and non-system messages.
	var system string
	var msgs []oaiMsg
	for _, m := range req.Messages {
		if m.Role == "system" {
			system = extractTextContent(m.Content)
		} else {
			msgs = append(msgs, m)
		}
	}

	// Convert messages. Tool result messages (role="tool") are folded into the
	// subsequent or current user message as tool_result content blocks, because
	// Anthropic does not have a standalone tool role.
	var antMsgs []antMsg
	var pendingToolResults []antBlock // tool_result blocks awaiting a user message

	flushPending := func() {
		if len(pendingToolResults) > 0 {
			antMsgs = append(antMsgs, antMsg{Role: "user", Content: pendingToolResults})
			pendingToolResults = nil
		}
	}

	for _, m := range msgs {
		switch m.Role {
		case "user":
			text := extractTextContent(m.Content)
			blocks := append(pendingToolResults, antBlock{Type: "text", Text: text})
			pendingToolResults = nil
			antMsgs = append(antMsgs, antMsg{Role: "user", Content: blocks})

		case "assistant":
			flushPending()
			var blocks []antBlock
			if text := extractTextContent(m.Content); text != "" {
				blocks = append(blocks, antBlock{Type: "text", Text: text})
			}
			for _, tc := range m.ToolCalls {
				input := json.RawMessage(tc.Function.Arguments)
				if len(input) == 0 {
					input = json.RawMessage(`{}`)
				}
				blocks = append(blocks, antBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Function.Name,
					Input: input,
				})
			}
			if len(blocks) == 0 {
				blocks = []antBlock{{Type: "text", Text: ""}}
			}
			antMsgs = append(antMsgs, antMsg{Role: "assistant", Content: blocks})

		case "tool":
			pendingToolResults = append(pendingToolResults, antBlock{
				Type:      "tool_result",
				ToolUseID: m.ToolCallID,
				Content:   extractTextContent(m.Content),
			})
		}
	}
	flushPending() // flush any trailing tool results with no following user message

	// Convert tools: unwrap the OpenAI {"type":"function","function":{...}} wrapper.
	var tools []antTool
	for _, t := range req.Tools {
		schema := t.Function.Parameters
		if len(schema) == 0 {
			schema = json.RawMessage(`{}`)
		}
		tools = append(tools, antTool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: schema,
		})
	}

	return json.Marshal(antRequest{
		Model:       model,
		MaxTokens:   maxTokens,
		System:      system,
		Messages:    antMsgs,
		Tools:       tools,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		TopP:        req.TopP,
	})
}

// TranslateResponse converts a non-streaming Anthropic Messages API response
// to OpenAI chat completion format. model is reported as-is in the output.
func TranslateResponse(anthropicResp []byte, model string) ([]byte, error) {
	var resp antResponse
	if err := json.Unmarshal(anthropicResp, &resp); err != nil {
		return nil, fmt.Errorf("parse anthropic response: %w", err)
	}

	var textBuf strings.Builder
	var toolCalls []oaiToolCall

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			textBuf.WriteString(block.Text)
		case "tool_use":
			args := string(block.Input)
			if args == "" || args == "null" {
				args = "{}"
			}
			toolCalls = append(toolCalls, oaiToolCall{
				ID:   block.ID,
				Type: "function",
				Function: oaiFunction{
					Name:      block.Name,
					Arguments: args,
				},
			})
		}
	}

	var content *string
	if s := textBuf.String(); s != "" {
		content = &s
	}

	return json.Marshal(oaiResponse{
		ID:      "chatcmpl-" + resp.ID,
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []oaiChoice{{
			Index: 0,
			Message: oaiMsgOut{
				Role:      "assistant",
				Content:   content,
				ToolCalls: toolCalls,
			},
			FinishReason: mapStopReason(resp.StopReason),
		}},
		Usage: oaiUsage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      resp.Usage.InputTokens + resp.Usage.OutputTokens,
		},
	})
}

// mapStopReason converts Anthropic stop_reason to OpenAI finish_reason.
func mapStopReason(reason string) string {
	switch reason {
	case "end_turn":
		return "stop"
	case "tool_use":
		return "tool_calls"
	case "max_tokens":
		return "length"
	default:
		return "stop"
	}
}

// extractTextContent parses an OpenAI content field, which can be a JSON
// string or an array of content parts, and returns the plain text.
func extractTextContent(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Type == "text" {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return ""
}
