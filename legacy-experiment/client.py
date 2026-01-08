import requests
import json

# Configuration
# Since network_mode: host is used in docker-compose, localhost works from host machine.
# If running inside the container, localhost also works.
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen3-4B"

def main():
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! Tell me a short joke."}
        ],
        "temperature": 0.7
    }

    print(f"Sending request to {API_URL}...")
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for 4xx/5xx errors
        
        result = response.json()
        
        # Print the full response for debugging
        # print(json.dumps(result, indent=2))
        
        # Extract and print just the content
        content = result['choices'][0]['message']['content']
        print("-" * 40)
        print("Response:")
        print(content)
        print("-" * 40)
        
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        if hasattr(e, 'response') and e.response is not None:
             print(f"Server response: {e.response.text}")

if __name__ == "__main__":
    main()

