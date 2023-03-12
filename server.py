import uvicorn
from fastapi import FastAPI, File, UploadFile
from functions import verify_attendence

app = FastAPI(
    title="Query PDF using LLM",
    description="Use LLM to search for information in PDF files",
    version="0.1",
)

@app.get("/", tags=['Home'], name='home')
async def root():
    return {"message": "AI attandence system v0.1. Go to <BASE_URL>/docs to see the API documentation."}

@app.post("/identify/",  name='Upload image file to verify face and detect IC number')
def create_db(file: UploadFile = File(...)):
    with open("img.jpeg", "wb") as f:
        f.write(file.file.read())
    file.file.close()
    result = verify_attendence("img.jpeg")
    return {"result": result}
        
if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)
    


