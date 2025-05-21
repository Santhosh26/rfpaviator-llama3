
from openai import OpenAI
# from src.compare_comp.refine_query import refine_query


def perplex(query, product):
    YOUR_API_KEY = "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"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an artificial intelligence assistant and you need to "
                "begin your responses with either 'Yes,' or 'No,' depending on sentiment of the response."
            ),
        },
        {
            "role": "user",
            "content": (
                ""
            ),
        },
    ]



    # Example usage
    ammended_product = "Does " + product + " "

    original_query = ammended_product + query
    
    # # refined_query = refine_query(original_query, product)
    # print(f"refined query: {original_query}" )

    # Update the 'content' for the user message with the refined query
    messages[1]["content"] = original_query

    client = OpenAI(api_key=YOUR_API_KEY, base_url="https://api.perplexity.ai")

    # chat completion without streaming
    response = client.chat.completions.create(
        model="mixtral-8x7b-instruct",
        messages=messages,
    )
    # Extracting the 'content' from the response
    # Assuming there is at least one choice in the response
    if response.choices:
        assistant_message = response.choices[0].message.content
        # print(f"Assistant's response: {assistant_message}")
        return assistant_message
    else:
        print("No response from the assistant.")
        return None


# print(perplex(query="The solution should support over 70 programming languages (include scripting languages) and provide IDE plugins for scanning and remediation.", product="codacy"))