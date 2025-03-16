from fastapi import FastAPI
from pydantic import BaseModel
from inference import load_model, generate_response

app = FastAPI()

model, tokenizer, device = load_model()

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def get_response(request: PromptRequest):
    response = generate_response(request.prompt, model, tokenizer, device)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)