from fastapi import FastAPI, File, UploadFile, APIRouter
from fastapi.responses import StreamingResponse
import cv2
import mediapipe as mp
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from ultralytics import YOLO
from mediapipe import solutions

router = APIRouter()

# Mediapipe pose estimation setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load YOLOv8 model
yolo_model = YOLO('yolov8n.pt')

FEEDBACK_SERVICE_URL = "https://pythonruleengine.onrender.com/check_posture/"  # Replace with actual URL


def get_keypoints(image):
    # Convert the PIL image to a NumPy array
    image_np = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format by default)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Use YOLO model to detect objects (including people)
    results = yolo_model(image_bgr)
    keypoints = []

    # Assuming the first detected person (if any) is at index 0
    if results and results[0].boxes.xyxy is not None:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            # Crop image to detected person region
            person_image = image_bgr[y1:y2, x1:x2]

            # Process cropped image with MediaPipe Pose
            pose_results = pose.process(person_image)

            if pose_results.pose_landmarks:
                keypoints = [
                    {"x": landmark.x, "y": landmark.y, "visibility": landmark.z}
                    for landmark in pose_results.pose_landmarks.landmark
                ]

    return {"landmarks": keypoints, "annotated_image": image_bgr}  # Return keypoints and image for feedback



@router.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    # Read image data from the uploaded file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # Extract keypoints and annotated image
    result = get_keypoints(image)
    keypoints = result["landmarks"]
    annotated_image = result["annotated_image"]


    # Send keypoints to external feedback service
    # Define the payload
    pose_input = {
        "landmarks": keypoints
    }
    response = requests.post(FEEDBACK_SERVICE_URL, json=pose_input)
    feedback = response.json().get("feedback", [])
    # Example of a print statement
    print("Feedback: ", feedback)
    # Add feedback to image
    for i, message in enumerate(feedback):
        cv2.putText(annotated_image, message, (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Convert image to response format
    processed_image = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    img_byte_arr = BytesIO()
    processed_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    print("img_byte_arr: ", img_byte_arr)
    return StreamingResponse(img_byte_arr, media_type="image/png")
