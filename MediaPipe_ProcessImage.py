from fastapi import FastAPI, File, UploadFile,APIRouter
from fastapi.responses import StreamingResponse
import cv2
import mediapipe as mp
import numpy as np
from io import BytesIO
from PIL import Image


router = APIRouter()

# Mediapipe pose estimation setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# Helper function to calculate angles between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


# Pose checks for Surya Namaskar
def check_hasta_uttanasana(landmarks):
    feedback = []
    left_elbow_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    )
    right_elbow_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
         landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
    )

    if left_elbow_angle < 160:
        feedback.append("Straighten your left arm.")
    if right_elbow_angle < 160:
        feedback.append("Straighten your right arm.")
    return feedback


def check_uttanasana(landmarks):
    feedback = []
    left_knee_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    )
    right_knee_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    )

    if left_knee_angle < 160:
        feedback.append("Straighten your left leg.")
    if right_knee_angle < 160:
        feedback.append("Straighten your right leg.")
    return feedback


def check_phalakasana(landmarks):
    feedback = []
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    body_angle = calculate_angle(shoulder, hip, ankle)
    if not (160 <= body_angle <= 180):
        feedback.append("Keep your body in a straight line.")

    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    if abs(left_wrist.x - left_shoulder.x) > 0.1:
        feedback.append("Move your left wrist under your shoulder.")
    return feedback


def analyze_pose(landmarks):
    if not landmarks:
        return ["Pose not detected."]

    landmarks = landmarks.landmark
    feedback = []

    # Surya Namaskar steps (simplified based on pose angles)
    # Identify specific steps based on landmark analysis
    # This could be extended by more specific checks for each step
    left_knee_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    )
    right_knee_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
        [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
    )

    if left_knee_angle < 160 and right_knee_angle < 160:
        feedback.append("You're in Uttanasana.")
    else:
        feedback.append("Pose not recognized yet.")

    return feedback


# Process the uploaded image
def analyzing_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = pose.process(image)
    feedback = []

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        feedback = analyze_pose(results.pose_landmarks)

        # Display feedback messages clearly on the image
        for i, message in enumerate(feedback):
            cv2.putText(image, message, (10, 30 + (i * 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Convert the processed image to a PIL Image for returning
    processed_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return processed_image

@router.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    # Read image data from the uploaded file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # Process the image
    processed_image = analyzing_image(image)

    # Convert the PIL image to BytesIO for streaming as response
    img_byte_arr = BytesIO()
    processed_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")