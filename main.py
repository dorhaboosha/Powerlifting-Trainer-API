import os

# ✅ Must be set BEFORE importing mediapipe (prevents EGL/GL GPU issues)
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import time
import threading
import shutil
import sqlite3
import json
import webbrowser
from email.message import EmailMessage
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import uvicorn
import aiosmtplib

from dotenv import load_dotenv
from openai import OpenAI

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, EmailStr

# -------------------------
# ✅ Env / runtime flags
# -------------------------
load_dotenv()

RUNNING_IN_DOCKER = os.getenv("RUNNING_IN_DOCKER", "0") == "1"
DB_PATH = os.getenv("DB_PATH", "Users.db")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./tmp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# ✅ OpenAI
# -------------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------------------
# ✅ SMTP Config (Gmail)
# -------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Powerlifting Coach")

# -------------------------
# ✅ FastAPI metadata (Docs polish)
# -------------------------
tags_metadata = [
    {
        "name": "Users",
        "description": "Register and fetch user profile data (BMI + stored PR history).",
    },
    {
        "name": "Records",
        "description": "Update user lifting records (deadlift / squat / bench press).",
    },
    {
        "name": "Video Analysis",
        "description": "Analyze uploaded videos or try real-time webcam analysis (local only).",
    }
]

app = FastAPI(
    title="Powerlifting Trainer Assistant API",
    description=(
        "A FastAPI backend that manages lifter profiles and provides AI coaching feedback.\n\n"
        "**Important limitations:**\n"
        "- Real-time webcam analysis works **only on local machine** (not Docker / cloud).\n"
        "- Uploaded video analysis works locally and in Docker.\n"
    ),
    version="1.0.0",
    openapi_tags=tags_metadata,
)

# -------------------------
# ✅ TTS (thread-safe)
# -------------------------
tts_engine = pyttsx3.init()
tts_lock = threading.Lock()


# -------------------------
# ✅ DB helpers
# -------------------------
def get_db_connection() -> sqlite3.Connection:
    """
    Create and return a SQLite DB connection using DB_PATH.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_users_table() -> None:
    """
    Ensure the Users table exists.
    """
    conn = get_db_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS Users (
            email TEXT PRIMARY KEY,
            height REAL NOT NULL,
            weight REAL NOT NULL,
            name TEXT NOT NULL,
            bmi REAL NOT NULL,
            deadlift TEXT NOT NULL,
            squat TEXT NOT NULL,
            bench_press TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


create_users_table()


# -------------------------
# ✅ Models
# -------------------------
class User(BaseModel):
    """User registration input model."""

    email: EmailStr
    height: float = Field(gt=0, description="Height in meters")
    weight: float = Field(gt=0, description="Weight in kilograms")
    name: str


def calculate_bmi(height: float, weight: float) -> float:
    """Calculate BMI rounded to 2 decimals."""
    return round(weight / (height**2), 2)


# -------------------------
# ✅ Docs UX
# -------------------------
@app.get("/", include_in_schema=False)
def root_redirect_to_docs():
    """Redirect root URL to Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health_check", include_in_schema=False)
def health_check():
    """Health check endpoint (hidden from docs)."""
    return {"status": "ok"}


# ============================================================
# ✅ NEW CLEAN ENDPOINTS (these appear in Swagger docs)
# ============================================================

@app.post(
    "/users/register",
    tags=["Users"],
    summary="Register a new user",
    description="Creates a user profile and stores BMI + empty record arrays (deadlift/squat/bench).",
)
async def register_user(user: User):
    if user.email == "user@example.com":
        raise HTTPException(status_code=400, detail="Invalid email address provided.")
    if user.height <= 0.5:
        raise HTTPException(status_code=400, detail="Invalid height provided.")
    if user.weight <= 10:
        raise HTTPException(status_code=400, detail="Invalid weight provided.")
    if not user.name.strip() or user.name == "string" or user.name[0].isspace():
        raise HTTPException(status_code=400, detail="Invalid name provided.")

    conn = get_db_connection()

    user_data = {
        "email": user.email,
        "height": user.height,
        "weight": user.weight,
        "name": user.name.strip(),
        "bmi": calculate_bmi(user.height, user.weight),
        "deadlift": json.dumps([]),
        "squat": json.dumps([]),
        "bench_press": json.dumps([]),
    }

    try:
        conn.execute(
            """
            INSERT INTO Users (email, height, weight, name, bmi, deadlift, squat, bench_press)
            VALUES (:email, :height, :weight, :name, :bmi, :deadlift, :squat, :bench_press)
            """,
            user_data,
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")
    conn.close()

    return {"message": "Welcome to our exercise program", "user": user_data}


@app.get(
    "/users/profile",
    tags=["Users"],
    summary="Get user profile by email",
    description="Fetches user profile (including record arrays) by email query parameter.",
)
async def get_user_profile(user_email: EmailStr):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM Users WHERE email = ?", (str(user_email),)).fetchone()
    conn.close()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        **user,
        "deadlift": json.loads(user["deadlift"]),
        "squat": json.loads(user["squat"]),
        "bench_press": json.loads(user["bench_press"]),
    }


@app.post(
    "/users/records/update",
    tags=["Records"],
    summary="Update user lifting records",
    description="Append new PR values to squat / bench press / deadlift arrays for a given user email.",
)
async def update_user_records(
    email: EmailStr,
    new_deadlift: float = None,
    new_squat: float = None,
    new_bench_press: float = None,
):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM Users WHERE email = ?", (str(email),)).fetchone()

    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="User not found")

    squats = json.loads(user["squat"])
    bench_presses = json.loads(user["bench_press"])
    deadlifts = json.loads(user["deadlift"])

    if new_squat is not None:
        if new_squat < 0:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid weight of squat provided.")
        squats.append(new_squat)

    if new_bench_press is not None:
        if new_bench_press < 0:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid weight of bench press provided.")
        bench_presses.append(new_bench_press)

    if new_deadlift is not None:
        if new_deadlift < 0:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid weight of deadlift provided.")
        deadlifts.append(new_deadlift)

    conn.execute(
        "UPDATE Users SET squat = ?, deadlift = ?, bench_press = ? WHERE email = ?",
        (json.dumps(squats), json.dumps(deadlifts), json.dumps(bench_presses), str(email)),
    )
    conn.commit()
    conn.close()

    return {"message": "User's records updated successfully"}


# ============================================================
# ✅ MediaPipe / Analysis helpers
# ============================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c) -> float:
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def analyze_squat(landmarks) -> float:
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    return calculate_angle(hip, knee, ankle)


def analyze_deadlift(landmarks) -> float:
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    return calculate_angle(shoulder, hip, knee)


def analyze_benchpress(landmarks) -> float:
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    return calculate_angle(wrist, elbow, shoulder)


def analyze_exercise_form(video_path: str, exercise_type: str):
    if video_path is None or not video_path.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid video format, please upload an .mp4 file.")
    if exercise_type not in ["squat", "deadlift", "benchpress"]:
        raise HTTPException(status_code=400, detail="Unsupported exercise type")

    cap = cv2.VideoCapture(video_path)
    angles = []
    min_angle_squat = float("inf")
    min_angle_benchpress = float("inf")
    saw_pose = False

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    saw_pose = True
                    landmarks = results.pose_landmarks.landmark

                    if exercise_type == "squat":
                        angle = analyze_squat(landmarks)
                        min_angle_squat = min(min_angle_squat, angle)
                    elif exercise_type == "deadlift":
                        angle = analyze_deadlift(landmarks)
                        angles.append(angle)
                    else:
                        angle = analyze_benchpress(landmarks)
                        min_angle_benchpress = min(min_angle_benchpress, angle)

                    if not RUNNING_IN_DOCKER:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        cv2.putText(frame, f"Angle: {angle:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Video Analysis", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during video processing: {e}")

    finally:
        cap.release()
        if not RUNNING_IN_DOCKER:
            cv2.destroyAllWindows()

    if not saw_pose:
        raise HTTPException(status_code=400, detail="No pose landmarks detected in the video. Please upload a clearer video.")

    if exercise_type == "squat":
        return min_angle_squat, exercise_type
    if exercise_type == "deadlift":
        return (sum(angles) / len(angles) if angles else 0), exercise_type
    return min_angle_benchpress, exercise_type


def chat_with_ai_video(final_angle: float, exercise_type: str) -> str:
    prompt = (
        f"I analyzed your {exercise_type} form.\n"
        f"Measured angle: {final_angle:.2f} degrees.\n"
        "Give me practical coaching feedback (form tips + 2-3 actionable improvements)."
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional powerlifting and strength training coach. "
                    "Give clear, practical feedback and safety tips."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content


def say_text(text: str) -> None:
    try:
        with tts_lock:
            tts_engine.say(text)
            tts_engine.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")


def process_video_real_time(duration: int, exercise_type: str) -> int:
    """
    Real-time webcam processing:
    - Only works locally (not in Docker/cloud)
    - If webcam isn't available, returns a friendly message via HTTPException
    """
    webcam_msg = (
        "There is no connection to the webcam (available only when running locally). "
        "Please upload a video and I will analyze it."
    )

    if RUNNING_IN_DOCKER:
        raise HTTPException(status_code=400, detail=webcam_msg)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail=webcam_msg)

    display_message_time = 0
    count = 0
    exercise_completed = False
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if exercise_type == "squat":
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle = calculate_angle(hip, knee, ankle)
                    threshold_down, threshold_up = 90, 90
                    label, voice = "SQUAT", "Good Squat"

                elif exercise_type == "deadlift":
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    angle = calculate_angle(shoulder, hip, knee)
                    threshold_down, threshold_up = 60, 60
                    label, voice = "DEADLIFT", "Good Deadlift"

                elif exercise_type == "benchpress":
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    angle = calculate_angle(wrist, elbow, shoulder)
                    threshold_down, threshold_up = 75, 75
                    label, voice = "BENCHPRESS", "Good Benchpress"
                else:
                    raise HTTPException(status_code=400, detail="Invalid exercise type")

                cv2.putText(frame_bgr, f"Angle: {angle:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame_bgr, f"{label} Count: {count}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                if angle < threshold_down and not exercise_completed:
                    display_message_time = time.time() + 5
                    exercise_completed = True
                    threading.Thread(target=say_text, args=(voice,), daemon=True).start()

                if angle > threshold_up and exercise_completed:
                    count += 1
                    exercise_completed = False

                if time.time() < display_message_time:
                    cv2.putText(frame_bgr, f"GOOD {label}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                mp_drawing.draw_landmarks(frame_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow(f"{exercise_type} Detection", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if (time.time() - start_time) > duration:
                break

    cap.release()
    cv2.destroyAllWindows()
    return count


async def email_sender(email: str, feedback_content: str) -> dict:
    """
    Send feedback email via SMTP (works locally/Docker where SMTP is allowed).
    If SMTP credentials are missing -> returns a message without failing the API.
    """
    if not SMTP_USER or not SMTP_PASS:
        return {"message": "Email not sent (missing SMTP_USER / SMTP_PASS in .env)"}

    msg = EmailMessage()
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_USER}>"
    msg["To"] = email
    msg["Subject"] = "Your Feedback on your exercise"

    msg.set_content(f"Your feedback:\n\n{feedback_content}")
    msg.add_alternative(
        f"""
        <div>
            <p><strong>Your feedback on your exercise is:</strong></p>
            <p>{feedback_content}</p>
        </div>
        """,
        subtype="html",
    )

    try:
        await aiosmtplib.send(
            msg,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            start_tls=True,
            username=SMTP_USER,
            password=SMTP_PASS,
        )
        return {"message": "Email sent successfully"}
    except Exception as e:
        print(f"SMTP email failed: {e}")
        return {"message": "Failed to send email"}


def chat_with_ai_video_real_time(good_count: int, duration: int, exercise_type: str) -> str:
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a professional powerlifting and strength training coach."},
            {"role": "user", "content": f"I completed {good_count} good {exercise_type}s in {duration} seconds. Give me feedback."},
        ],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content


@app.post(
    "/video/process",
    tags=["Video Analysis"],
    summary="Analyze uploaded video OR run real-time webcam (local only)",
    description=(
        "If you upload an `.mp4` file, the API analyzes it with MediaPipe and returns AI coaching feedback.\n\n"
        "If you do NOT upload a video, you can request real-time analysis by setting `Duration_in_real_time > 0`.\n"
        "**Real-time webcam mode works only locally** (not Docker/cloud)."
    ),
)
async def video_process(
    Video: UploadFile = File(None),
    Exercise_type: str = Form(..., description="One of: squat, deadlift, benchpress"),
    Duration_in_real_time: Optional[int] = Form(0, description="Real-time duration in seconds (local only)"),
    email: EmailStr = Form(..., description="Recipient email for the feedback"),
):
    if Exercise_type.lower() not in ["squat", "deadlift", "benchpress"]:
        raise HTTPException(status_code=400, detail="Invalid exercise type.")

    # 1) Uploaded video path
    if Video is not None:
        temp_video_path = os.path.join(UPLOAD_DIR, f"temp_{Video.filename}")
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(Video.file, buffer)

        final_angle, exercise = analyze_exercise_form(temp_video_path, Exercise_type.lower())
        feedback = chat_with_ai_video(final_angle, exercise)

        os.remove(temp_video_path)

        email_result = await email_sender(str(email), feedback)
        return {
            "feedback": feedback,
            "message": "Uploaded video analysis complete.",
            "email_status": email_result["message"],
        }

    # 2) Real-time (webcam)
    if Duration_in_real_time and Duration_in_real_time > 0:
        try:
            good_count = process_video_real_time(Duration_in_real_time, Exercise_type.lower())
        except HTTPException as e:
            return {"message": e.detail}

        ai_feedback = chat_with_ai_video_real_time(good_count, Duration_in_real_time, Exercise_type.lower())
        email_result = await email_sender(str(email), ai_feedback)

        return {
            "feedback": ai_feedback,
            "message": "Real-time analysis complete.",
            "email_status": email_result["message"],
        }

    return {"message": "Please specify a positive duration or upload a video."}


# ============================================================
# ✅ OLD ENDPOINTS (kept for compatibility, hidden from docs)
# ============================================================

@app.post("/Registration", include_in_schema=False)
async def old_register_user(user: User):
    return await register_user(user)


@app.get("/Show User", include_in_schema=False)
async def old_get_user(user_email: EmailStr):
    return await get_user_profile(user_email)


@app.post("/Update Record Weight", include_in_schema=False)
async def old_update_weight(
    email: EmailStr,
    new_deadlift: float = None,
    new_squat: float = None,
    new_bench_press: float = None,
):
    return await update_user_records(
        email=email,
        new_deadlift=new_deadlift,
        new_squat=new_squat,
        new_bench_press=new_bench_press,
    )


@app.post("/Video Processing", include_in_schema=False)
async def old_video_processing(
    Video: UploadFile = File(None),
    Exercise_type: str = Form(...),
    Duration_in_real_time: Optional[int] = Form(0),
    email: EmailStr = Form(...),
):
    return await video_process(Video, Exercise_type, Duration_in_real_time, email)


if __name__ == "__main__":
    host = "0.0.0.0" if RUNNING_IN_DOCKER else "127.0.0.1"
    port = 8000

    # ✅ Auto-open docs only when running locally (not Docker)
    if not RUNNING_IN_DOCKER:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{port}/docs")).start()

    uvicorn.run(app, host=host, port=port)
