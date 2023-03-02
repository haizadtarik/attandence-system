import uvicorn
from fastapi import FastAPI, File, UploadFile
from functions import detect_ic, face_detection, face_recognition

app = FastAPI(
    title="Query PDF using LLM",
    description="Use LLM to search for information in PDF files",
    version="0.1",
)

@app.get("/", tags=['Home'], name='home')
async def root():
    return {"message": "AI attandence system v0.1. Go to <BASE_URL>/docs to see the API documentation."}

@app.post("/upload/", name='Upload image file to detect face and save in database')
def create_db(file: UploadFile = File(...)):
    with open(file.filename,'wb') as image:
        image.write(file.file.read())
        image.close()
    face_detection(file.filename, 'ssd')
    return {"results": f"Image of {file.filename.split('.')[0]}uploaded successfully"}

@app.post("/identify/",  name='Upload image file to recognize face and detect IC number')
def create_db(file: UploadFile = File(...)):
    with open('image.jpg','wb') as image:
        image.write(file.file.read())
        image.close()
    identified_user = face_recognition('image.jpg', 'db', 'Facenet')
    if identified_user == "Unknown":
        return {"results": f"{identified_user}"}
    else:
        ic_num = detect_ic('image.jpg')
        return {"results": f"User: {identified_user}, IC: {ic_num}"}

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)
    


