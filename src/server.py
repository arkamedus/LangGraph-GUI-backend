# server.py

import os
from datetime import datetime
import httpx
from typing import Dict
import asyncio

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from ServerTee import ServerTee
from process_handler import ProcessHandler
from FileTransmit import file_router

# log name as today's date in YYYY-MM-DD format
today_date = datetime.now().strftime("%Y-%m-%d")
# Create log file path dynamically based on the date
log_file_path = f"log/{today_date}.log"
# Initialize ServerTee with the dynamically generated log file path
tee = ServerTee(log_file_path)
# Print the log file path for reference
print(log_file_path)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Dictionary to store ProcessHandler instances per user
handlers = {}


@app.post('/chatbot/{username}')
async def process_string(request: Request, username: str):
    # Get the JSON data from the request
    data = await request.json()
    input_string = data.get('input_string', '')
    llm_model = data.get('llm_model', '')  # Default to 'gemma2' if not provided
    api_key = data.get('api_key', '')

    # Process the string using the dynamically provided llm_model and api_key
    result = await ChatBot(get_llm(llm_model, api_key), input_string)

    # Return the result as JSON
    return JSONResponse(content={'result': result})


@app.post('/run/{username}')
async def run_script(request: Request, username: str):
    user_workspace = os.path.join("workspace", username)

    data = await request.json()
    llm_model = data.get('llm_model', '')
    api_key = data.get('api_key', '')

    command = [
        "python", "../../run_graph.py",
        "--llm", llm_model,
        "--key", api_key
    ]

    # Get or create a handler for the user
    if username not in handlers:
        handlers[username] = ProcessHandler()
    
    handler = handlers[username]
    # start process in background
    async for output in handler.get_stream():
        try:
            # If output is already a dict, just dump it as JSON.
            if isinstance(output, dict):
                yield f"data: {json.dumps(output)}\n\n"
                break  # assume dict signals final status
            else:
                # Try to parse output string as JSON.
                try:
                    parsed = json.loads(output)
                except json.JSONDecodeError:
                    # Fallback: use ast.literal_eval to convert single quotes to a dict.
                    parsed = ast.literal_eval(output)
                yield f"data: {json.dumps(parsed)}\n\n"
        except Exception as e:
            error_response = {"status": "error", "message": str(e)}
            yield f"data: {json.dumps(error_response)}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

@app.get('/status/{username}')
async def check_status(username: str):
    # Check if the handler exists and retrieve its status
    if username in handlers:
        status = await handlers[username].status()  # Note: status() is an async function
        return {"running": status["is_running"]}  # Make sure to return running status
    return {"running": False}


# Include file router
app.include_router(file_router)

# Run the app using Uvicorn
if __name__ == "__main__":
    import uvicorn

    backend_port = int(os.environ.get("BACKEND_PORT", 5000))  # Default to 5000 if not set
    uvicorn.run(app, host="0.0.0.0", port=backend_port, reload=True)