from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Root works"}

@app.get("/test")
def test():
    return {"message": "Test works"}

@app.post("/generate")
def generate():
    return {"message": "Generate works"}
