package main

import (
	"context"
	"fmt"

	"github.com/danielmesquitta/ai-tool-calling/gpt"
	"github.com/danielmesquitta/ai-tool-calling/gpt/openai"
)

func main() {
	var openaiGPT gpt.GPT = openai.NewOpenAI("your-api-key-here")

	systemMessage := gpt.Message{
		Role:    gpt.RoleSystem,
		Content: "You are a financial counselor agent. Help users understand their financial data by answering questions about their transactions, budgets, and account balance using the available tools.",
	}

	userMessage := gpt.Message{
		Role:    gpt.RoleUser,
		Content: "Could you give me an overview of my transactions and current balance?",
	}

	jsonTransactionCategories := `[
		"groceries",
		"utilities",
		"entertainment",
		"transportation",
		"healthcare",
		"education"
	]`

	openaiGPT.Completion(
		context.Background(),
		[]gpt.Message{
			systemMessage,
			userMessage,
		},
		gpt.WithTools([]gpt.Tool{
			{
				Name:        "listTransactions",
				Description: "List all user transactions filtered by date and category.",
				Func:        listTransactions("your-user-id"),
				Args: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"start_date": map[string]any{
							"type":        "string",
							"description": "The start date for filtering balances (RFC3339 format).",
						},
						"end_date": map[string]any{
							"type":        "string",
							"description": "The end date for filtering balances (RFC3339 format).",
						},
						"category_id": map[string]any{
							"type": "array",
							"items": map[string]any{
								"type": "string",
							},
							"description": fmt.Sprintf(
								"The categories to filter transactions by (use empty array to list all). Here is all possible categories: %s",
								jsonTransactionCategories,
							),
						},
					},
					"required": []string{
						"start_date",
						"end_date",
						"category_id",
					},
					"additionalProperties": false,
				},
			},
		}),
	)
}

func listTransactions(
	userID string,
) func(ctx context.Context, args map[string]any) (string, error) {
	return func(ctx context.Context, args map[string]any) (string, error) {
		startDate := args["start_date"].(string)
		endDate := args["end_date"].(string)
		return fmt.Sprintf(
			"The user %s has 5 transactions this month totaling $1500. Your current balance is $5000. (Filtered from %s to %s)",
			userID,
			startDate,
			endDate,
		), nil
	}
}
