# src/fastapi_llama_server.py
from fastapi import FastAPI, HTTPException
from transformers import pipeline
import torch

app = FastAPI()

# Global variable to store the pipeline
pipe = None

def load_model():
    global pipe
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/chat/")
async def chat(messages: list):
    if pipe is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")

    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    return {"response": outputs[0]["generated_text"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)