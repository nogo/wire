package anthropic

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

const testModel = "claude-3-5-sonnet-20241022"

// --- TranslateRequest tests ---

func TestTranslateRequest_Basic(t *testing.T) {
	req := `{"model":"gpt-4","messages":[{"role":"user","content":"Hello"}]}`
	got, err := TranslateRequest([]byte(req), testModel)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var ant antRequest
	if err := json.Unmarshal(got, &ant); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if ant.Model != testModel {
		t.Errorf("model = %q, want %q", ant.Model, testModel)
	}
	if ant.MaxTokens != 8192 {
		t.Errorf("max_tokens = %d, want 8192 (default)", ant.MaxTokens)
	}
	if len(ant.Messages) != 1 {
		t.Fatalf("messages len = %d, want 1", len(ant.Messages))
	}
	if ant.Messages[0].Role != "user" {
		t.Errorf("role = %q, want user", ant.Messages[0].Role)
	}
	if len(ant.Messages[0].Content) != 1 || ant.Messages[0].Content[0].Text != "Hello" {
		t.Errorf("content = %+v, want text block with Hello", ant.Messages[0].Content)
	}
}

func TestTranslateRequest_MaxTokensPreserved(t *testing.T) {
	req := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"max_tokens":500}`
	got, err := TranslateRequest([]byte(req), testModel)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var ant antRequest
	json.Unmarshal(got, &ant) //nolint:errcheck
	if ant.MaxTokens != 500 {
		t.Errorf("max_tokens = %d, want 500", ant.MaxTokens)
	}
}

func TestTranslateRequest_SystemMessage(t *testing.T) {
	req := `{"model":"gpt-4","messages":[{"role":"system","content":"Be helpful"},{"role":"user","content":"Hi"}]}`
	got, err := TranslateRequest([]byte(req), testModel)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var ant antRequest
	json.Unmarshal(got, &ant) //nolint:errcheck

	if ant.System != "Be helpful" {
		t.Errorf("system = %q, want %q", ant.System, "Be helpful")
	}
	// System message must not appear in messages array.
	for _, m := range ant.Messages {
		if m.Role == "system" {
			t.Error("system role found in messages array")
		}
	}
	if len(ant.Messages) != 1 {
		t.Errorf("messages len = %d, want 1", len(ant.Messages))
	}
}

func TestTranslateRequest_ToolCalls(t *testing.T) {
	req := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "What's the weather?"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\":\"NYC\"}"}}
			]},
			{"role": "tool", "tool_call_id": "call_abc123", "content": "Sunny, 75F"}
		]
	}`
	got, err := TranslateRequest([]byte(req), testModel)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var ant antRequest
	json.Unmarshal(got, &ant) //nolint:errcheck

	// Expect: user, assistant, user(tool_result)
	if len(ant.Messages) != 3 {
		t.Fatalf("messages len = %d, want 3", len(ant.Messages))
	}

	// assistant message should have tool_use block
	asst := ant.Messages[1]
	if asst.Role != "assistant" {
		t.Errorf("msg[1].role = %q, want assistant", asst.Role)
	}
	if len(asst.Content) != 1 || asst.Content[0].Type != "tool_use" {
		t.Fatalf("msg[1].content = %+v, want one tool_use block", asst.Content)
	}
	if asst.Content[0].ID != "call_abc123" {
		t.Errorf("tool_use.id = %q, want call_abc123", asst.Content[0].ID)
	}
	if asst.Content[0].Name != "get_weather" {
		t.Errorf("tool_use.name = %q, want get_weather", asst.Content[0].Name)
	}

	// Tool result should be folded into a user message
	toolResultMsg := ant.Messages[2]
	if toolResultMsg.Role != "user" {
		t.Errorf("msg[2].role = %q, want user", toolResultMsg.Role)
	}
	if len(toolResultMsg.Content) != 1 || toolResultMsg.Content[0].Type != "tool_result" {
		t.Fatalf("msg[2].content = %+v, want one tool_result block", toolResultMsg.Content)
	}
	if toolResultMsg.Content[0].ToolUseID != "call_abc123" {
		t.Errorf("tool_result.tool_use_id = %q, want call_abc123", toolResultMsg.Content[0].ToolUseID)
	}
	if toolResultMsg.Content[0].Content != "Sunny, 75F" {
		t.Errorf("tool_result.content = %q, want Sunny, 75F", toolResultMsg.Content[0].Content)
	}
}

func TestTranslateRequest_ToolCallIDPreserved(t *testing.T) {
	req := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "Run it"},
			{"role": "assistant", "content": null, "tool_calls": [
				{"id": "call_XYZ789", "type": "function", "function": {"name": "run", "arguments": "{}"}}
			]},
			{"role": "tool", "tool_call_id": "call_XYZ789", "content": "done"}
		]
	}`
	got, err := TranslateRequest([]byte(req), testModel)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var ant antRequest
	json.Unmarshal(got, &ant) //nolint:errcheck

	if ant.Messages[1].Content[0].ID != "call_XYZ789" {
		t.Errorf("tool_use.id = %q, want call_XYZ789", ant.Messages[1].Content[0].ID)
	}
	if ant.Messages[2].Content[0].ToolUseID != "call_XYZ789" {
		t.Errorf("tool_result.tool_use_id = %q, want call_XYZ789", ant.Messages[2].Content[0].ToolUseID)
	}
}

func TestTranslateRequest_ToolDefinitions(t *testing.T) {
	req := `{
		"model": "gpt-4",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}}}]
	}`
	got, err := TranslateRequest([]byte(req), testModel)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var ant antRequest
	json.Unmarshal(got, &ant) //nolint:errcheck

	if len(ant.Tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(ant.Tools))
	}
	if ant.Tools[0].Name != "get_weather" {
		t.Errorf("tool.name = %q, want get_weather", ant.Tools[0].Name)
	}
	if ant.Tools[0].Description != "Get weather" {
		t.Errorf("tool.description = %q, want Get weather", ant.Tools[0].Description)
	}
	var schema map[string]any
	json.Unmarshal(ant.Tools[0].InputSchema, &schema) //nolint:errcheck
	if schema["type"] != "object" {
		t.Errorf("input_schema.type = %v, want object", schema["type"])
	}
}

func TestTranslateRequest_ToolResultsAtEnd(t *testing.T) {
	// Tool results at the end (no following user message) should flush to a user message.
	req := `{
		"model": "gpt-4",
		"messages": [
			{"role": "user", "content": "go"},
			{"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
			{"role": "tool", "tool_call_id": "call_1", "content": "result"}
		]
	}`
	got, err := TranslateRequest([]byte(req), testModel)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var ant antRequest
	json.Unmarshal(got, &ant) //nolint:errcheck

	last := ant.Messages[len(ant.Messages)-1]
	if last.Role != "user" {
		t.Errorf("last message role = %q, want user", last.Role)
	}
	if len(last.Content) != 1 || last.Content[0].Type != "tool_result" {
		t.Errorf("last message content = %+v, want tool_result block", last.Content)
	}
}

func TestTranslateRequest_EmptyMessages(t *testing.T) {
	req := `{"model":"gpt-4","messages":[]}`
	_, err := TranslateRequest([]byte(req), testModel)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestTranslateRequest_StreamPassthrough(t *testing.T) {
	req := `{"model":"gpt-4","messages":[{"role":"user","content":"hi"}],"stream":true}`
	got, err := TranslateRequest([]byte(req), testModel)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	var ant antRequest
	json.Unmarshal(got, &ant) //nolint:errcheck
	if !ant.Stream {
		t.Error("stream = false, want true")
	}
}

// --- TranslateResponse tests ---

func TestTranslateResponse_Basic(t *testing.T) {
	resp := `{
		"id": "msg_01XF",
		"type": "message",
		"role": "assistant",
		"content": [{"type": "text", "text": "Hello there!"}],
		"model": "claude-3-5-sonnet-20241022",
		"stop_reason": "end_turn",
		"usage": {"input_tokens": 10, "output_tokens": 5}
	}`
	got, err := TranslateResponse([]byte(resp), "my-backend")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var oai oaiResponse
	json.Unmarshal(got, &oai) //nolint:errcheck

	if oai.Object != "chat.completion" {
		t.Errorf("object = %q, want chat.completion", oai.Object)
	}
	if oai.Model != "my-backend" {
		t.Errorf("model = %q, want my-backend", oai.Model)
	}
	if len(oai.Choices) != 1 {
		t.Fatalf("choices len = %d, want 1", len(oai.Choices))
	}
	if oai.Choices[0].Message.Role != "assistant" {
		t.Errorf("role = %q, want assistant", oai.Choices[0].Message.Role)
	}
	if oai.Choices[0].Message.Content == nil || *oai.Choices[0].Message.Content != "Hello there!" {
		t.Errorf("content = %v, want Hello there!", oai.Choices[0].Message.Content)
	}
	if oai.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", oai.Choices[0].FinishReason)
	}
	if oai.Usage.PromptTokens != 10 || oai.Usage.CompletionTokens != 5 || oai.Usage.TotalTokens != 15 {
		t.Errorf("usage = %+v, want {10 5 15}", oai.Usage)
	}
}

func TestTranslateResponse_ToolCalls(t *testing.T) {
	resp := `{
		"id": "msg_02",
		"type": "message",
		"role": "assistant",
		"content": [{"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"location": "NYC"}}],
		"model": "claude-3-5-sonnet-20241022",
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 20, "output_tokens": 10}
	}`
	got, err := TranslateResponse([]byte(resp), testModel)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var oai oaiResponse
	json.Unmarshal(got, &oai) //nolint:errcheck

	if oai.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls", oai.Choices[0].FinishReason)
	}
	if oai.Choices[0].Message.Content != nil {
		t.Errorf("content = %v, want nil (null) when only tool_calls", oai.Choices[0].Message.Content)
	}
	tc := oai.Choices[0].Message.ToolCalls
	if len(tc) != 1 {
		t.Fatalf("tool_calls len = %d, want 1", len(tc))
	}
	if tc[0].ID != "toolu_abc" {
		t.Errorf("tool_call.id = %q, want toolu_abc", tc[0].ID)
	}
	if tc[0].Function.Name != "get_weather" {
		t.Errorf("tool_call.function.name = %q, want get_weather", tc[0].Function.Name)
	}
	// Arguments should be the JSON representation of the input object.
	var args map[string]any
	json.Unmarshal([]byte(tc[0].Function.Arguments), &args) //nolint:errcheck
	if args["location"] != "NYC" {
		t.Errorf("arguments.location = %v, want NYC", args["location"])
	}
}

func TestTranslateResponse_StopReasonMapping(t *testing.T) {
	cases := []struct{ in, want string }{
		{"end_turn", "stop"},
		{"tool_use", "tool_calls"},
		{"max_tokens", "length"},
		{"stop_sequence", "stop"},
		{"", "stop"},
	}
	for _, c := range cases {
		got := mapStopReason(c.in)
		if got != c.want {
			t.Errorf("mapStopReason(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// --- TranslateStream tests ---

func TestTranslateStream_Text(t *testing.T) {
	stream := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	w := httptest.NewRecorder()
	TranslateStream(strings.NewReader(stream), w, "my-model")

	body := w.Body.String()

	// Should have data: [...] lines and end with [DONE]
	if !strings.Contains(body, "data: ") {
		t.Error("no SSE data lines in output")
	}
	if !strings.HasSuffix(strings.TrimRight(body, "\n"), "data: [DONE]") {
		t.Errorf("output does not end with [DONE], got: %q", body)
	}

	// Collect all SSE chunks
	chunks := parseOAISSE(t, body)

	// First chunk should have role=assistant
	if len(chunks) == 0 {
		t.Fatal("no chunks")
	}
	first := chunks[0]
	if first.Choices[0].Delta.Role != "assistant" {
		t.Errorf("first chunk role = %q, want assistant", first.Choices[0].Delta.Role)
	}

	// Find chunks with text content
	var texts []string
	for _, c := range chunks {
		if c.Choices[0].Delta.Content != nil && *c.Choices[0].Delta.Content != "" {
			texts = append(texts, *c.Choices[0].Delta.Content)
		}
	}
	if strings.Join(texts, "") != "Hello world" {
		t.Errorf("text = %q, want %q", strings.Join(texts, ""), "Hello world")
	}

	// Last data chunk (before [DONE]) should have finish_reason
	dataChunks := parseOAISSEChunks(t, body)
	last := dataChunks[len(dataChunks)-1]
	if last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "stop" {
		t.Errorf("last chunk finish_reason = %v, want stop", last.Choices[0].FinishReason)
	}

	// All chunks should carry our model name
	for _, c := range chunks {
		if c.Model != "my-model" {
			t.Errorf("chunk model = %q, want my-model", c.Model)
		}
	}
}

func TestTranslateStream_ToolCall(t *testing.T) {
	stream := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"usage":{"input_tokens":15,"output_tokens":0}}}` + "\n\n" +
		"event: content_block_start\n" +
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_xyz","name":"get_weather"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"loc"}}` + "\n\n" +
		"event: content_block_delta\n" +
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ation\":\"NYC\"}"}}` + "\n\n" +
		"event: content_block_stop\n" +
		`data: {"type":"content_block_stop","index":0}` + "\n\n" +
		"event: message_delta\n" +
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":5}}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	w := httptest.NewRecorder()
	TranslateStream(strings.NewReader(stream), w, "my-model")

	body := w.Body.String()
	chunks := parseOAISSEChunks(t, body)

	// Find the initial tool call chunk (has id and name)
	var foundHeader bool
	var foundArgs bool
	for _, c := range chunks {
		if len(c.Choices[0].Delta.ToolCalls) == 0 {
			continue
		}
		tc := c.Choices[0].Delta.ToolCalls[0]
		if tc.ID == "toolu_xyz" && tc.Function.Name == "get_weather" {
			foundHeader = true
			if tc.Function.Arguments != nil && *tc.Function.Arguments == "" {
				// initial args chunk is empty string
			}
		}
		if tc.Function.Arguments != nil && strings.Contains(*tc.Function.Arguments, "loc") {
			foundArgs = true
		}
	}
	if !foundHeader {
		t.Error("no tool call header chunk with id=toolu_xyz and name=get_weather")
	}
	if !foundArgs {
		t.Error("no tool call argument chunks")
	}

	// finish_reason should be tool_calls
	last := chunks[len(chunks)-1]
	if last.Choices[0].FinishReason == nil || *last.Choices[0].FinishReason != "tool_calls" {
		t.Errorf("last finish_reason = %v, want tool_calls", last.Choices[0].FinishReason)
	}
}

func TestTranslateStream_PingIgnored(t *testing.T) {
	stream := "event: message_start\n" +
		`data: {"type":"message_start","message":{"id":"msg_3","type":"message","role":"assistant","content":[],"model":"claude","stop_reason":null,"usage":{"input_tokens":5,"output_tokens":0}}}` + "\n\n" +
		"event: ping\n" +
		`data: {"type":"ping"}` + "\n\n" +
		"event: message_stop\n" +
		`data: {"type":"message_stop"}` + "\n\n"

	w := httptest.NewRecorder()
	TranslateStream(strings.NewReader(stream), w, "my-model")

	body := w.Body.String()
	if !strings.Contains(body, "data: [DONE]") {
		t.Error("expected [DONE] in output")
	}
	// ping should not appear as a data chunk
	chunks := parseOAISSEChunks(t, body)
	// Should have: message_start chunk + message_stop [DONE] (no ping chunk)
	for _, c := range chunks {
		raw, _ := json.Marshal(c)
		if strings.Contains(string(raw), "ping") {
			t.Error("ping event leaked into output")
		}
	}
}

// --- helpers ---

// parseOAISSE returns all chunks (excluding [DONE]) from an OpenAI SSE body.
func parseOAISSE(t *testing.T, body string) []oaiChunk {
	t.Helper()
	return parseOAISSEChunks(t, body)
}

func parseOAISSEChunks(t *testing.T, body string) []oaiChunk {
	t.Helper()
	var chunks []oaiChunk
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := line[6:]
		if data == "[DONE]" {
			continue
		}
		var c oaiChunk
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			t.Fatalf("parse chunk %q: %v", data, err)
		}
		chunks = append(chunks, c)
	}
	return chunks
}
