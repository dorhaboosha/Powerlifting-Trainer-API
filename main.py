import os

# ✅ Must be set BEFORE importing mediapipe (prevents EGL/GL GPU issues)
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"

import time
import mediapipe as mp
import pyttsx3
import threading
import shutil
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field, EmailStr
import sqlite3
import json
import cv2
import numpy as np
from typing import Optional
import uvicorn

# ✅ Gmail SMTP async sender
from email.message import EmailMessage
import aiosmtplib

# -------------------------
# ✅ Docker/headless flags
# -------------------------
RUNNING_IN_DOCKER = os.getenv("RUNNING_IN_DOCKER", "0") == "1"

# DB path (so we can mount it later)
DB_PATH = os.getenv("DB_PATH", "Users.db")

# Upload temp directory (so Docker has a safe place to write)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./tmp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI
app = FastAPI()

# Load env + OpenAI
load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ✅ SMTP Config (Gmail)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Powerlifting Coach")

# -------------------------
# ✅ Thread-safe TTS
# -------------------------
tts_engine = pyttsx3.init()
tts_lock = threading.Lock()


def get_db_connection():
    """
    Establishes and returns a connection to the SQLite database.
    Uses DB_PATH so Docker can persist data via mounted volume.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def create_users_table():
    """
    Creates a 'Users' table in the SQLite database if it doesn't already exist.
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


class User(BaseModel):
    email: EmailStr
    height: float = Field(gt=0, description="Height in meters")
    weight: float = Field(gt=0, description="Weight in kilograms")
    name: str


def calculate_bmi(height: float, weight: float) -> float:
    return round(weight / (height**2), 2)


@app.post("/Registration")
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
        "name": user.name,
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


@app.get("/Show User")
async def get_user(user_email: EmailStr):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM Users WHERE email = ?", (user_email,)).fetchone()
    conn.close()

    if user:
        return {
            **user,
            "deadlift": json.loads(user["deadlift"]),
            "squat": json.loads(user["squat"]),
            "bench_press": json.loads(user["bench_press"]),
        }
    raise HTTPException(status_code=404, detail="User not found")


@app.post("/Update Record Weight")
async def update_weight(
    email: EmailStr,
    new_deadlift: float = None,
    new_squat: float = None,
    new_bench_press: float = None,
):
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM Users WHERE email = ?", (email,)).fetchone()

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
        updated_squat_json = json.dumps(squats)
    else:
        updated_squat_json = user["squat"]

    if new_bench_press is not None:
        if new_bench_press < 0:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid weight of bench press provided.")
        bench_presses.append(new_bench_press)
        updated_benchpress_json = json.dumps(bench_presses)
    else:
        updated_benchpress_json = user["bench_press"]

    if new_deadlift is not None:
        if new_deadlift < 0:
            conn.close()
            raise HTTPException(status_code=400, detail="Invalid weight of deadlift provided.")
        deadlifts.append(new_deadlift)
        updated_deadlift_json = json.dumps(deadlifts)
    else:
        updated_deadlift_json = user["deadlift"]

    conn.execute(
        "UPDATE Users SET squat = ?, deadlift = ?, bench_press = ? WHERE email = ?",
        (updated_squat_json, updated_deadlift_json, updated_benchpress_json, email),
    )
    conn.commit()
    conn.close()

    return {"message": "User's records updated successfully"}


# MediaPipe init
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def analyze_squat(landmarks):
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    return calculate_angle(hip, knee, ankle)


def analyze_deadlift(landmarks):
    shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
    ]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    return calculate_angle(shoulder, hip, knee)


def analyze_benchpress(landmarks):
    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    shoulder = [
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
    ]
    return calculate_angle(wrist, elbow, shoulder)


def analyze_exercise_form(video_path, exercise_type):
    if video_path is None or not video_path.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid video format, please upload an .mp4 file.")
    if exercise_type not in ["squat", "deadlift", "benchpress"]:
        raise HTTPException(status_code=400, detail="Unsupported exercise type")

    cap = cv2.VideoCapture(video_path)
    angles = []
    min_angle_squat = float("inf")
    min_angle_benchpress = float("inf")

    try:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
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

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # ✅ Only show UI locally (not in Docker)
                    if not RUNNING_IN_DOCKER:
                        cv2.putText(
                            frame,
                            f"Angle: {angle:.2f} degrees",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        cv2.imshow("Video Analysis", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during video processing: {e}")

    finally:
        cap.release()
        if not RUNNING_IN_DOCKER:
            cv2.destroyAllWindows()

    if exercise_type == "squat":
        return min_angle_squat, exercise_type
    if exercise_type == "deadlift":
        return (sum(angles) / len(angles) if angles else 0), exercise_type
    return min_angle_benchpress, exercise_type


def chat_with_ai_video(final_angle, exercise_type):
    if exercise_type == "squat":
        prompt = f"I am analyzing your squat form and it seems your knee angle is {final_angle} degrees. This is good for squatting!"
    elif exercise_type == "deadlift":
        prompt = f"I am analyzing your deadlift form and it seems your shoulder angle is {final_angle} degrees. Try to keep your back straighter."
    elif exercise_type == "benchpress":
        prompt = f"I am analyzing your bench press form and it seems your elbow angle is {final_angle} degrees. Make sure to keep your elbows stable."
    else:
        raise HTTPException(status_code=400, detail="Invalid exercise type")

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a professional powerlifting and strength training coach. "
                    f"Provide guidance on improving performance in {exercise_type}. "
                    "Provide explanations and tips clearly."
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


def process_video_real_time(duration: int, exercise_type: str):
    """
    Real-time webcam processing:
    - Not available in Docker/Azure.
    - If webcam can't be opened locally, return a friendly message.
    """
    webcam_msg = (
        "There is no connection to the webcam (available only when running locally). "
        "Please upload a video and I will analyze it."
    )

    # ✅ Block real-time mode in Docker/Azure
    if RUNNING_IN_DOCKER:
        raise HTTPException(status_code=400, detail=webcam_msg)

    cap = cv2.VideoCapture(0)

    # ✅ Local machine but no webcam / blocked permissions
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
            frame_bgr_for_display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if exercise_type == "squat":
                    hip = [
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    ]
                    knee = [
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                    ]
                    ankle = [
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
                    ]
                    angle = calculate_angle(hip, knee, ankle)
                    threshold_down, threshold_up = 90, 90
                    label = "SQUAT"
                    voice = "Good Squat"

                elif exercise_type == "deadlift":
                    shoulder = [
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                    ]
                    hip = [
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y,
                    ]
                    knee = [
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y,
                    ]
                    angle = calculate_angle(shoulder, hip, knee)
                    threshold_down, threshold_up = 60, 60
                    label = "DEADLIFT"
                    voice = "Good Deadlift"

                elif exercise_type == "benchpress":
                    wrist = [
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                    ]
                    elbow = [
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                    ]
                    shoulder = [
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                    ]
                    angle = calculate_angle(wrist, elbow, shoulder)
                    threshold_down, threshold_up = 75, 75
                    label = "BENCHPRESS"
                    voice = "Good Benchpress"
                else:
                    raise HTTPException(status_code=400, detail="Invalid exercise type")

                # UI (local only)
                cv2.putText(
                    frame_bgr_for_display,
                    f"Angle: {angle:.2f}",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_bgr_for_display,
                    f"{label} Count: {count}",
                    (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2,
                )

                if angle < threshold_down and not exercise_completed:
                    display_message_time = time.time() + 5
                    exercise_completed = True
                    threading.Thread(target=say_text, args=(voice,), daemon=True).start()

                if angle > threshold_up and exercise_completed:
                    count += 1
                    exercise_completed = False

                if time.time() < display_message_time:
                    cv2.putText(
                        frame_bgr_for_display,
                        f"GOOD {label}",
                        (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )

                mp_drawing.draw_landmarks(
                    frame_bgr_for_display, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            cv2.imshow(f"{exercise_type} Detection", frame_bgr_for_display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if (time.time() - start_time) > duration:
                break

    cap.release()
    cv2.destroyAllWindows()
    return count


async def email_sender(email: str, feedback_content: str) -> dict:
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


def chat_with_ai_video_real_time(good_count: int, duration: int, exercise_type: str):
    if exercise_type == "squat":
        angle = "knee"
    elif exercise_type == "deadlift":
        angle = "back"
    elif exercise_type == "benchpress":
        angle = "elbow"
    else:
        raise HTTPException(status_code=400, detail="Invalid exercise type")

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a professional powerlifting and strength training coach."},
            {
                "role": "user",
                "content": f"I completed {good_count} good {exercise_type}s in {duration} seconds. Give me feedback.",
            },
        ],
        model="gpt-4o-mini",
    )
    return chat_completion.choices[0].message.content


@app.post("/Video Processing")
async def process_video_If_you_dont_upload_anything_it_will_analyze_video_in_real_time(
    Video: UploadFile = File(None),
    Exercise_type: str = Form(...),
    Duration_in_real_time: Optional[int] = Form(0),
    email: EmailStr = Form(...),
):
    if Exercise_type and Exercise_type.lower() not in ["squat", "deadlift", "benchpress"]:
        raise HTTPException(status_code=400, detail="Invalid exercise type.")

    # 1) Uploaded video path
    if Video and Exercise_type:
        temp_video_path = os.path.join(UPLOAD_DIR, f"temp_{Video.filename}")
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(Video.file, buffer)

        final_angle, exercise = analyze_exercise_form(temp_video_path, Exercise_type.lower())
        feedback = chat_with_ai_video(final_angle, exercise)

        os.remove(temp_video_path)

        email_result = await email_sender(email, feedback) if email else None

        return {
            "feedback": feedback,
            "message": "Uploaded video analysis complete.",
            "email_status": (email_result["message"] if email_result else "No email requested"),
        }

    # 2) Real-time (webcam)
    if Duration_in_real_time and Duration_in_real_time > 0:
        try:
            good_count = process_video_real_time(Duration_in_real_time, Exercise_type.lower())
        except HTTPException as e:
            # ✅ Return friendly message instead of hard-failing
            return {"message": e.detail}

        ai_feedback = chat_with_ai_video_real_time(good_count, Duration_in_real_time, Exercise_type.lower())
        email_result = await email_sender(email, ai_feedback) if email else None

        return {
            "feedback": ai_feedback,
            "message": "Real-time analysis complete.",
            "email_status": (email_result["message"] if email_result else "No email requested"),
        }

    return {"message": "Please specify a positive duration or upload a video."}


if __name__ == "__main__":
    # NOTE: When running in Docker you should bind to 0.0.0.0
    uvicorn.run(app, host="0.0.0.0" if RUNNING_IN_DOCKER else "127.0.0.1", port=8000)
