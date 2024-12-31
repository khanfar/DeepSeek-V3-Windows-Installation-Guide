import argparse
import asyncio
import uvicorn
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from typing import AsyncGenerator
import json
from fastapi.responses import StreamingResponse, JSONResponse
import threading
import os

app = FastAPI(title="DeepSeek Windows Server")

def create_arg_parser():
    parser = argparse.ArgumentParser(description="DeepSeek Windows Server")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    return parser

# Initialize model and tokenizer globally
model = None
tokenizer = None

def initialize_model(model_path: str, trust_remote_code: bool):
    global model, tokenizer
    print(f"Loading model from {model_path}...")
    print("This may take a few minutes depending on your system...")
    
    try:
        # Disable quantization warning
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            device_map="auto"
        )
        print("Model loaded successfully!")
        print(f"Model is loaded on device: {model.device}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have enough GPU memory")
        print("2. Verify the model path is correct")
        print("3. Check if model conversion was successful")
        raise e

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        temperature = body.get("temperature", 0.7)
        max_tokens = body.get("max_tokens", 100)
        
        # Build prompt from messages
        prompt = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        # Set up streaming
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )
        
        # Start generation in a separate thread
        thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        if stream:
            async def response_generator() -> AsyncGenerator[str, None]:
                for new_text in streamer:
                    response = {
                        "choices": [{
                            "delta": {"role": "assistant", "content": new_text},
                            "finish_reason": None
                        }],
                        "object": "chat.completion.chunk"
                    }
                    yield f"data: {json.dumps(response)}\n\n"
                
                # Send the final message
                yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                response_generator(),
                media_type="text/event-stream"
            )
        else:
            # For non-streaming, collect the entire response
            response_text = ""
            for new_text in streamer:
                response_text += new_text
            
            return JSONResponse({
                "choices": [{
                    "message": {"role": "assistant", "content": response_text.strip()},
                    "finish_reason": "stop"
                }],
                "object": "chat.completion"
            })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    # Initialize the model
    initialize_model(args.model, args.trust_remote_code)
    
    # Print server info
    print("\nServer Information:")
    print("-" * 50)
    print(f"Server running at: http://127.0.0.1:{args.port}")
    print(f"Health check: http://127.0.0.1:{args.port}/health")
    print(f"Chat endpoint: http://127.0.0.1:{args.port}/v1/chat/completions")
    print("-" * 50)
    
    # Run the server
    uvicorn.run(app, host="127.0.0.1", port=args.port)

if __name__ == "__main__":
    main()
