import re

def process_cb_openai(callbacks_list):
    # print(f"cb: {callbacks_list}")
    total_tokens_used = 0
    total_cost = 0.0

    for cb in callbacks_list:
        # Extract tokens used and cost from the callback output
        tokens_used_match = re.search(r"Tokens Used: (\d+)", cb)
        cost_match = re.search(r"Total Cost \(USD\): \$(\d+\.\d+)", cb)

        if tokens_used_match and cost_match:
            total_tokens_used += int(tokens_used_match.group(1))
            total_cost += float(cost_match.group(1))
    
    total_token_cost_message= f"Total tokens used for RFP response: {total_tokens_used} and total cost is: USD{total_cost:.4f}"
    return total_token_cost_message