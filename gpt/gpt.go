package gpt

import "context"

type Role = string

const (
	RoleAssistant Role = "assistant"
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
)

type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}

type ToolFunc func(ctx context.Context, args map[string]any) (string, error)

type Tool struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Func        ToolFunc
	Args        map[string]any `json:"args"`
}

type Model = string

const (
	ModelO3Mini Model = "o3-mini"
	Model4oMini Model = "gpt-4o-mini"
)

type Options struct {
	Temperature float32 `json:"temperature"`
	Seed        *int    `json:"seed"`
	Tools       []Tool  `json:"tools"`
	Model       Model   `json:"model"`
}

type Option func(*Options)

func WithTemperature(temperature float32) Option {
	return func(o *Options) {
		o.Temperature = temperature
	}
}

func WithSeed(seed int) Option {
	return func(o *Options) {
		o.Seed = &seed
	}
}

func WithTools(tools []Tool) Option {
	return func(o *Options) {
		o.Tools = tools
	}
}

func WithModel(model Model) Option {
	return func(o *Options) {
		o.Model = model
	}
}

type GPT interface {
	Completion(
		ctx context.Context,
		messages []Message,
		options ...Option,
	) (*Message, error)
}
