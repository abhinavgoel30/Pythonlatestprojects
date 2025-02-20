from fastapi import FastAPI
import MediaPipe_ProcessImage
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include the routes from endpoints.py
app.include_router(MediaPipe_ProcessImage.router)

@app.get("/")
def home():
    return {"message": "Hello, FastAPI!"}



