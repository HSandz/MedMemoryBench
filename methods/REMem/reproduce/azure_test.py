import os
from openai import AzureOpenAI

if __name__ == "__main__":
    assert os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_AD_TOKEN")
    assert os.getenv("OPENAI_API_VERSION")
    model_name = 'gpt-4o-mini'
    client = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), max_retries=5)

    # IMAGE_PATH = "YOUR_IMAGE_PATH"
    # encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

    chat_prompt = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You're an AI assistant. Print any named entities in JSON format, e.g., {\"named_entities\": [\"apple\", \"banana\"]}"
                }
            ]
        }
    ]

    messages = chat_prompt
    response_format = {"type": "json_object"}

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=800,
        temperature=0.7,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None,
        stream=False,
        response_format=response_format,
    )

    print(completion.to_json())
