import openai
import time

def test_model():
    # Initialize the OpenAI client with our local server
    client = openai.Client(
        base_url="http://127.0.0.1:30000/v1",
        api_key="EMPTY"  # Not needed for local server
    )
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Write a short poem about coding in Python."}
    ]
    
    print("Sending test request to the server...")
    try:
        # Make the request
        response = client.chat.completions.create(
            model="deepseek-ai/deepseek-llm-7b-base",
            messages=messages,
            temperature=0.7,
            max_tokens=100
        )
        
        # Print the response
        print("\nModel Response:")
        print("-" * 50)
        print(response.choices[0].message.content)
        print("-" * 50)
        print("\nTest completed successfully!")
        
    except Exception as e:
        print("\nError occurred while testing:")
        print(f"{e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure the server is running (python windows_server.py)")
        print("2. Check if the server URL is correct (http://127.0.0.1:30000)")
        print("3. Verify that the model was loaded successfully")

if __name__ == "__main__":
    test_model()
