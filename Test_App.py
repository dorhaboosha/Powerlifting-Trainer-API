from dotenv import load_dotenv
from fastapi.testclient import TestClient
import pytest
import sqlite3
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
import main
import json

from main import (
    app,
    get_db_connection,
    calculate_bmi,
    analyze_squat,
    analyze_exercise_form,
    analyze_benchpress,
    analyze_deadlift,
    mp_pose,
    chat_with_ai_video,
    say_text,
    email_sender,
    calculate_angle,
    update_weight,
)

load_dotenv()

client = TestClient(app)


@pytest.fixture
def mock_db_connection(mocker):
    mocker.patch("sqlite3.connect", return_value=MagicMock(sqlite3.Connection))


def test_get_db_connection(mock_db_connection):
    conn = get_db_connection()
    assert conn is not None


def test_calculate_bmi():
    height = 1.75
    weight = 75
    expected_bmi = 24.49
    bmi = calculate_bmi(height, weight)
    assert bmi == expected_bmi


def pre_cleanup():
    """Ensure the test user does not exist before running the test."""
    conn = get_db_connection()
    conn.execute("DELETE FROM Users WHERE email = ?", ("test@example.com",))
    conn.commit()
    conn.close()


def post_cleanup():
    """Delete the test user after the test."""
    conn = get_db_connection()
    conn.execute("DELETE FROM Users WHERE email = ?", ("test@example.com",))
    conn.commit()
    conn.close()


def test_register_user():
    pre_cleanup()

    response = client.post(
        "/Registration",
        json={
            "email": "test@example.com",
            "height": 1.75,
            "weight": 75,
            "name": "Test User",
        },
    )

    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to our exercise program"

    post_cleanup()


def test_get_user_not_found():
    test_email = "NotExists@example.com"
    response = client.get(f"/Show User?user_email={test_email}")
    assert response.status_code == 404


def test_calculate_angle_degrees():
    a = [0, 0]
    b = [1, 0]
    c = [1, 1]
    expected_angle = 90.0

    calculated_angle = calculate_angle(a, b, c)
    assert calculated_angle == expected_angle


def mock_landmark(x, y):
    lm = MagicMock()
    lm.x = x
    lm.y = y
    return lm


def test_analyze_squat():
    hip = mock_landmark(0.5, 0.5)
    knee = mock_landmark(0.5, 0.6)
    ankle = mock_landmark(0.5, 0.7)

    landmarks = [None] * 33
    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] = hip
    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value] = knee
    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value] = ankle

    expected_angle = calculate_angle((hip.x, hip.y), (knee.x, knee.y), (ankle.x, ankle.y))
    calculated_angle = analyze_squat(landmarks)

    assert calculated_angle == pytest.approx(expected_angle)


def test_analyze_deadlift():
    shoulder = mock_landmark(0.5, 0.4)
    hip = mock_landmark(0.5, 0.5)
    knee = mock_landmark(0.5, 0.6)

    landmarks = [None] * 33
    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = shoulder
    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value] = hip
    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value] = knee

    expected_angle = calculate_angle((shoulder.x, shoulder.y), (hip.x, hip.y), (knee.x, knee.y))
    calculated_angle = analyze_deadlift(landmarks)

    assert calculated_angle == pytest.approx(expected_angle)


def test_analyze_benchpress():
    wrist = mock_landmark(0.4, 0.5)
    elbow = mock_landmark(0.5, 0.5)
    shoulder = mock_landmark(0.6, 0.5)

    landmarks = [None] * 33
    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value] = wrist
    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value] = elbow
    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = shoulder

    expected_angle = calculate_angle((wrist.x, wrist.y), (elbow.x, elbow.y), (shoulder.x, shoulder.y))
    calculated_angle = analyze_benchpress(landmarks)

    assert calculated_angle == pytest.approx(expected_angle)


@pytest.fixture
def mock_openai_client(mocker):
    mocker.patch(
        "main.client.chat.completions.create",
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Mocked AI response"))]),
    )


def test_analyze_exercise_form_invalid_input(mock_openai_client):
    with pytest.raises(HTTPException) as e:
        analyze_exercise_form(None, "squat")
    assert e.value.status_code == 400
    assert "Invalid video format" in str(e.value.detail)

    with pytest.raises(HTTPException) as e2:
        analyze_exercise_form("video.mp4", "unsupported_exercise")
    assert e2.value.status_code == 400


def test_chat_with_ai_video(mock_openai_client):
    final_angle = 45
    exercise_type = "squat"
    response = chat_with_ai_video(final_angle, exercise_type)
    assert response == "Mocked AI response"


@patch("pyttsx3.init")
def test_say_text(mock_init):
    text = "This is a test message."
    say_text(text)
    mock_init.assert_called_once()
    mock_init.return_value.say.assert_called_once_with(text)
    mock_init.return_value.runAndWait.assert_called_once()


@patch("main.mp_drawing.draw_landmarks")
@patch("main.cv2.putText")
@patch("main.cv2.cvtColor")
@patch("main.cv2.VideoCapture")
@patch("main.cv2.imshow")
@patch("main.cv2.waitKey")
@patch("main.cv2.destroyAllWindows")
@patch("main.mp_pose.Pose")
@patch("main.calculate_angle")
def test_process_video_real_time(
    mock_calculate_angle,
    mock_Pose,
    mock_destroyAllWindows,
    mock_waitKey,
    mock_imshow,
    mock_VideoCapture,
    mock_cvtColor,
    mock_putText,
    mock_draw_landmarks,
):
    mock_calculate_angle.return_value = 45.0

    mock_putText.side_effect = lambda *args, **kwargs: None
    mock_draw_landmarks.side_effect = lambda *args, **kwargs: None

    mock_VideoCapture_instance = mock_VideoCapture.return_value
    mock_VideoCapture_instance.isOpened.return_value = True
    mock_frame = MagicMock()
    mock_frame.shape = (100, 100, 3)
    mock_VideoCapture_instance.read.return_value = (True, mock_frame)

    mock_results = MagicMock()
    mock_results.pose_landmarks.landmark = [MagicMock() for _ in range(33)]
    mock_Pose.return_value.process.return_value = mock_results
    mock_Pose.return_value.POSE_CONNECTIONS = [(15, 21)]

    duration = 1
    exercise_type = "squat"
    count = main.process_video_real_time(duration, exercise_type)

    assert count == 0
    assert mock_VideoCapture_instance.isOpened.called
    assert mock_VideoCapture_instance.read.called
    assert mock_Pose.called
    assert mock_destroyAllWindows.called
    assert mock_cvtColor.called
    assert mock_putText.called
    assert mock_draw_landmarks.called


# -------------------------
# ✅ Email tests for Resend (testing mode)
# -------------------------
@patch("main.resend.Emails.send")
def test_email_sender_success(mock_send, monkeypatch):
    monkeypatch.setattr(main, "RESEND_API_KEY", "re_test_key")
    monkeypatch.setattr(main, "RESEND_FROM", "Acme <onboarding@resend.dev>")

    # ✅ Make the test deterministic (don’t rely on your real .env)
    monkeypatch.setenv("RESEND_TEST_TO", "test@example.com")

    mock_send.return_value = {"id": "email_123"}

    email = "test@example.com"
    feedback_content = "This is a test feedback"
    result = email_sender(email, feedback_content)

    assert result == {"message": "Email sent successfully"}

    args, _ = mock_send.call_args
    sent_payload = args[0]

    assert sent_payload["from"] == "Acme <onboarding@resend.dev>"
    assert sent_payload["to"] == [email]
    assert sent_payload["subject"] == "Your Feedback on your exercise"
    assert "This is a test feedback" in sent_payload["html"]


@patch("main.resend.Emails.send")
def test_email_sender_failure(mock_send, monkeypatch):
    monkeypatch.setattr(main, "RESEND_API_KEY", "re_test_key")
    monkeypatch.setattr(main, "RESEND_FROM", "Acme <onboarding@resend.dev>")

    # ✅ Allow this recipient in testing mode
    monkeypatch.setenv("RESEND_TEST_TO", "test@example.com")

    mock_send.return_value = {}  # no id => failure

    email = "test@example.com"
    feedback_content = "This is a test feedback"
    result = email_sender(email, feedback_content)

    assert result == {"message": "Failed to send email"}


@patch("main.resend.Emails.send")
def test_email_sender_blocks_non_test_recipient(mock_send, monkeypatch):
    monkeypatch.setattr(main, "RESEND_API_KEY", "re_test_key")
    monkeypatch.setattr(main, "RESEND_FROM", "Acme <onboarding@resend.dev>")

    # Only this is allowed
    monkeypatch.setenv("RESEND_TEST_TO", "allowed@example.com")

    email = "not-allowed@example.com"
    feedback_content = "This is a test feedback"
    result = email_sender(email, feedback_content)

    assert result["message"].startswith("Email not sent.")
    mock_send.assert_not_called()


@pytest.mark.asyncio
async def test_update_weight_success(mocker):
    conn_mock = mocker.MagicMock()
    mocker.patch("main.get_db_connection", return_value=conn_mock)

    email = "test@example.com"
    user_row = {"email": email, "squat": "[]", "bench_press": "[]", "deadlift": "[]"}
    conn_mock.execute.return_value.fetchone.return_value = user_row

    new_squat = 100.0
    new_bench_press = 150.0
    new_deadlift = 200.0

    response = await update_weight(
        email=email,
        new_squat=new_squat,
        new_bench_press=new_bench_press,
        new_deadlift=new_deadlift,
    )

    assert response == {"message": "User's records updated successfully"}

    conn_mock.execute.assert_any_call("SELECT * FROM Users WHERE email = ?", (email,))
    conn_mock.execute.assert_any_call(
        "UPDATE Users SET squat = ?, deadlift = ?, bench_press = ? WHERE email = ?",
        (json.dumps([new_squat]), json.dumps([new_deadlift]), json.dumps([new_bench_press]), email),
    )
    conn_mock.commit.assert_called_once()
    conn_mock.close.assert_called_once()
