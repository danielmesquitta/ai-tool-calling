package openai

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sync"

	"github.com/sashabaranov/go-openai"
	"golang.org/x/sync/errgroup"

	"github.com/danielmesquitta/ai-tool-calling/gpt"
)

type OpenAI struct {
	Client *openai.Client
}

func NewOpenAI(
	apiKey string,
) *OpenAI {
	client := openai.NewClient(apiKey)

	return &OpenAI{
		Client: client,
	}
}

func (o *OpenAI) Completion(
	ctx context.Context,
	messages []gpt.Message,
	options ...gpt.Option,
) (*gpt.Message, error) {
	opts := gpt.Options{}
	for _, opt := range options {
		opt(&opts)
	}

	params := o.prepareParams(messages, opts)

	toolsByName := make(map[string]gpt.Tool)
	for _, tool := range opts.Tools {
		toolsByName[tool.Name] = tool
	}

	const maxAttempts = 3
	for range maxAttempts {
		completion, err := o.Client.CreateChatCompletion(ctx, params)
		if err != nil {
			return nil, err
		}
		if len(completion.Choices) == 0 {
			return nil, (fmt.Errorf("no choices returned"))
		}

		choice := completion.Choices[0].Message
		if len(choice.ToolCalls) == 0 {
			message := gpt.Message{
				Role:    gpt.RoleAssistant,
				Content: choice.Content,
			}
			return &message, nil
		}

		params.Messages = append(params.Messages, choice)

		err = o.processToolCalls(ctx, choice.ToolCalls, toolsByName, &params)
		if err != nil {
			return nil, err
		}
	}

	return nil, errors.New("max attempts reached without a valid response")
}

func (o *OpenAI) processToolCalls(
	ctx context.Context,
	toolCalls []openai.ToolCall,
	toolsByName map[string]gpt.Tool,
	params *openai.ChatCompletionRequest,
) error {
	g, subCtx := errgroup.WithContext(ctx)
	mu := sync.Mutex{}

	for _, tc := range toolCalls {
		g.Go(func() error {
			tool, ok := toolsByName[tc.Function.Name]
			if !ok {
				slog.Error("tool not found", "tool_id", tc.Function.Name)
				return nil
			}

			var args map[string]any
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				slog.Error(
					"failed to unmarshal tool args",
					"tool_id", tc.ID,
					"error", err,
				)
				return nil
			}

			slog.Info("Tool call", "name", tc.Function.Name, "args", args)

			toolMessage, err := tool.Func(subCtx, args)
			if err != nil {
				slog.Error(
					"tool function call failed",
					"tool_id", tc.ID,
					"error", err,
				)
				return nil
			}

			mu.Lock()
			params.Messages = append(
				params.Messages,
				openai.ChatCompletionMessage{
					Role:       openai.ChatMessageRoleTool,
					Content:    toolMessage,
					Name:       tc.Function.Name,
					ToolCallID: tc.ID,
				},
			)
			mu.Unlock()

			return nil
		})
	}

	return g.Wait()
}

func (o *OpenAI) prepareParams(
	messages []gpt.Message,
	opts gpt.Options,
) openai.ChatCompletionRequest {
	openAIMessages := o.parseMessages(messages)
	openAITools := o.parseTools(opts.Tools)

	model := cmp.Or(opts.Model, openai.O3Mini)

	param := openai.ChatCompletionRequest{
		Tools:    openAITools,
		Messages: openAIMessages,
		Model:    model,
	}

	if opts.Temperature != 0 {
		param.Temperature = opts.Temperature
	}

	if opts.Seed != nil {
		param.Seed = opts.Seed
	}

	return param
}

func (o *OpenAI) parseTools(
	tools []gpt.Tool,
) []openai.Tool {
	openAITools := []openai.Tool{}
	for _, tool := range tools {
		openAITool := openai.Tool{
			Type: openai.ToolTypeFunction,
			Function: &openai.FunctionDefinition{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Args,
				Strict:      true,
			},
		}

		openAITools = append(openAITools, openAITool)
	}

	return openAITools
}

func (o *OpenAI) parseMessages(
	messages []gpt.Message,
) []openai.ChatCompletionMessage {
	openAIMessages := []openai.ChatCompletionMessage{}
	for _, message := range messages {
		msg, err := o.parseMessage(message)
		if err != nil {
			slog.Error("failed to map gpt message", "error", err)
			continue
		}

		openAIMessages = append(
			openAIMessages,
			*msg,
		)
	}

	return openAIMessages
}

func (o *OpenAI) parseMessage(
	message gpt.Message,
) (*openai.ChatCompletionMessage, error) {
	var msg = openai.ChatCompletionMessage{
		Content: message.Content,
	}

	switch message.Role {
	case gpt.RoleAssistant:
		msg.Role = openai.ChatMessageRoleAssistant

	case gpt.RoleSystem:
		msg.Role = openai.ChatMessageRoleSystem

	case gpt.RoleUser:
		msg.Role = openai.ChatMessageRoleUser

	default:
		return nil, (fmt.Errorf("invalid role %s", message.Role))
	}

	return &msg, nil
}

var _ gpt.GPT = (*OpenAI)(nil)
