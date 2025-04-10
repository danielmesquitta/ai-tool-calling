# AI Tool Calling Example

A Go example project demonstrating how to use OpenAI's function calling capabilities with a clean abstraction layer.

## Overview

This project shows how to:

- Create a modular GPT client abstraction
- Implement OpenAI's function/tool calling feature
- Build a financial assistant bot that can fetch transaction data

## Installation

```bash
go mod download
```

## Usage

1. Add your OpenAI API key in `main.go`:

```go
openaiGPT := openai.NewOpenAI("your-api-key-here")
```

2. Run the example:

```bash
go run main.go
```

## Project Structure

- `gpt/`
  - `gpt.go` - Main GPT interface and types
  - `openai/`
    - `openai.go` - OpenAI client implementation

## Features

- Clean abstraction over OpenAI's API
- Support for function/tool calling
- Concurrent tool execution
- Configurable options (temperature, seed, model)
- Error handling and retries

## Example

The example implements a financial assistant that can:

- Query transaction history
- Filter by date ranges
- Filter by transaction categories
- Show account balances

Sample code:

```go
openaiGPT.Completion(
    context.Background(),
    []gpt.Message{
        {Role: gpt.RoleSystem, Content: "You are a financial counselor..."},
        {Role: gpt.RoleUser, Content: "Show me my transactions..."},
    },
    gpt.WithTools([]gpt.Tool{
        // Tool definitions...
    }),
)
```

## License

This project is MIT licensed.
