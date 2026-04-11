"""
Pytest suite for the Powerlifting Trainer Assistant API (``main`` module).

Rough layout (matches the order of tests below):

    1. **HTTP + DB** - ``TestClient`` against the FastAPI ``app``; DB access is often
       patched via ``mock_db_connection`` so CI never depends on a real ``Users.db``.
    2. **Pure helpers** - BMI and geometry (``calculate_angle``) need no I/O.
    3. **Pose helpers** - ``analyze_*`` functions use synthetic landmark lists (index =
       MediaPipe landmark id) instead of loading video.
    4. **OpenAI** - ``mock_openai_client`` stubs ``main.client.chat.completions.create``
       so tests never call the network.
    5. **Webcam path** - OpenCV and MediaPipe are patched; ``RUNNING_IN_DOCKER`` may be
       forced off so ``process_video_real_time`` exercises the happy path without a camera.
    6. **Email** - ``aiosmtplib.send`` is patched; ``SMTP_USER`` / ``SMTP_PASS`` on
       ``main`` are set so ``email_sender`` does not short-circuit early.
    7. **Records update** - ``get_db_connection`` returns a mock; ``execute`` uses
       ``side_effect`` so SELECT and UPDATE get distinct cursor mocks.

From the project root, run ``pytest Test_App.py -v``.

``load_dotenv()`` loads your ``.env`` for any test that still touches real config;
most tests isolate behavior with mocks or in-memory-friendly paths.
"""

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
    update_user_records,
    InvalidVideoFormatError,
    UnsupportedExerciseError,
)

load_dotenv()

# Shared ASGI client: same app instance as production, but requests never bind a real port.
client = TestClient(app)


# =============================================================================
# Database wiring and HTTP smoke tests
# =============================================================================


@pytest.fixture
def mock_db_connection(mocker):
    """
    Replace ``main.get_db_connection`` with a ``MagicMock`` connection.

    Route handlers call ``get_db_connection`` on the ``main`` module, so patching
    ``main.get_db_connection`` keeps registration tests off disk while still
    exercising FastAPI validation and response shape.
    """
    mocker.patch("main.get_db_connection", return_value=MagicMock(spec=sqlite3.Connection))


def test_get_db_connection(mock_db_connection):
    """get_db_connection returns a non-None connection (mock when patched)."""
    conn = main.get_db_connection()
    assert conn is not None
    main.get_db_connection.assert_called_once()


def test_calculate_bmi():
    """``calculate_bmi`` uses weight / height^2 rounded to two decimals (SI units)."""
    height = 1.75
    weight = 75
    expected_bmi = 24.49
    bmi = calculate_bmi(height, weight)
    assert bmi == expected_bmi


def test_register_user(mock_db_connection):
    """POST /users/register with valid payload creates user and returns welcome message (uses mocked DB)."""
    response = client.post(
        "/users/register",
        json={
            "email": "test@example.com",
            "height": 1.75,
            "weight": 75,
            "name": "Test User",
        },
    )

    assert response.status_code == 200
    assert response.json()["message"] == "Welcome to our exercise program"


def test_get_user_not_found():
    """
    ``GET /users/profile`` with a non-existent ``user_email`` yields 404.

    Uses the real DB path unless you add a similar patch fixture; keep the email
    obviously synthetic to avoid colliding with local dev data.
    """
    test_email = "NotExists@example.com"
    response = client.get(f"/users/profile?user_email={test_email}")
    assert response.status_code == 404


# =============================================================================
# Geometry: calculate_angle (no MediaPipe)
# =============================================================================


def test_calculate_angle_degrees():
    """Right angle at ``b`` for an L-shaped triple of 2D points in the plane."""
    a = [0, 0]
    b = [1, 0]
    c = [1, 1]
    expected_angle = 90.0

    calculated_angle = calculate_angle(a, b, c)
    assert calculated_angle == expected_angle


# =============================================================================
# Pose landmark helpers (synthetic 33-element landmark lists)
# =============================================================================


def mock_landmark(x, y):
    """Minimal stand-in for one MediaPipe landmark (normalized x, y only)."""
    lm = MagicMock()
    lm.x = x
    lm.y = y
    return lm


def test_analyze_squat():
    """``analyze_squat`` must agree with ``calculate_angle`` on the same three points."""
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
    """``analyze_deadlift`` uses shoulder-hip-knee; compare to explicit ``calculate_angle``."""
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
    """``analyze_benchpress`` uses wrist-elbow-shoulder; compare to ``calculate_angle``."""
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


# =============================================================================
# Video analysis validation and OpenAI (network-free)
# =============================================================================


@pytest.fixture
def mock_openai_client(mocker):
    """
    Stub the OpenAI client so ``chat_with_ai_video`` returns predictable text.

    Patches ``main.client`` (module-level singleton), not a local import inside
    the function under test.
    """
    mocker.patch(
        "main.client.chat.completions.create",
        return_value=MagicMock(choices=[MagicMock(message=MagicMock(content="Mocked AI response"))]),
    )


def test_analyze_exercise_form_invalid_input(mock_openai_client):
    """
    ``analyze_exercise_form`` raises domain exceptions (not ``HTTPException``).

    Routes translate these to HTTP status codes; this test calls the pure function
    directly. The OpenAI fixture is unused here but keeps the suite consistent
    if you merge video tests later.
    """
    with pytest.raises(InvalidVideoFormatError) as e:
        analyze_exercise_form(None, "squat")
    assert "Invalid video format" in str(e.value)

    with pytest.raises(UnsupportedExerciseError):
        analyze_exercise_form("video.mp4", "unsupported_exercise")


def test_chat_with_ai_video(mock_openai_client):
    """End-to-end string from ``chat_with_ai_video`` equals the stubbed completion content."""
    final_angle = 45
    exercise_type = "squat"
    response = chat_with_ai_video(final_angle, exercise_type)
    assert response == "Mocked AI response"


# =============================================================================
# Text-to-speech and real-time webcam (heavy deps mocked)
# =============================================================================


@patch("main.tts_engine")
def test_say_text(mock_tts_engine):
    """Patch ``main.tts_engine`` (not ``pyttsx3.init``) because ``say_text`` uses the module singleton."""
    text = "This is a test message."
    say_text(text)
    mock_tts_engine.say.assert_called_once_with(text)
    mock_tts_engine.runAndWait.assert_called_once()


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
    monkeypatch,
):
    """
    Exercise ``process_video_real_time`` without a camera or GUI.

    Patches are applied bottom-to-top in the decorator order (innermost first
    argument). ``Pose`` must mock ``__enter__`` because production code uses
    ``with mp_pose.Pose(...) as pose``.
    """
    # Otherwise the handler returns 400 immediately in Docker-oriented CI envs.
    monkeypatch.setattr(main, "RUNNING_IN_DOCKER", False)

    mock_calculate_angle.return_value = 45.0

    mock_putText.side_effect = lambda *args, **kwargs: None
    mock_draw_landmarks.side_effect = lambda *args, **kwargs: None

    mock_VideoCapture_instance = mock_VideoCapture.return_value
    # Exit after one iteration so the test does not depend on real wall-clock time
    mock_VideoCapture_instance.isOpened.side_effect = [True, False]
    mock_frame = MagicMock()
    mock_frame.shape = (100, 100, 3)
    mock_VideoCapture_instance.read.return_value = (True, mock_frame)

    mock_results = MagicMock()
    mock_results.pose_landmarks.landmark = [MagicMock() for _ in range(33)]
    mock_pose_instance = mock_Pose.return_value.__enter__.return_value
    mock_pose_instance.process.return_value = mock_results
    mock_pose_instance.POSE_CONNECTIONS = [(15, 21)]

    duration = 1  # Unused for loop exit here; ``isOpened`` side_effect ends the loop first
    exercise_type = "squat"
    count = main.process_video_real_time(duration, exercise_type)

    assert count == 0
    assert mock_VideoCapture_instance.isOpened.called
    assert mock_VideoCapture_instance.read.called
    mock_VideoCapture_instance.release.assert_called_once()
    assert mock_Pose.called
    assert mock_destroyAllWindows.called
    assert mock_cvtColor.called
    assert mock_putText.called
    assert mock_draw_landmarks.called


def test_process_video_real_time_blocks_in_docker(monkeypatch):
    """Docker flag should surface ``HTTPException`` 400 before opening device index 0."""
    monkeypatch.setattr(main, "RUNNING_IN_DOCKER", True)
    with pytest.raises(HTTPException) as e:
        main.process_video_real_time(1, "squat")
    assert e.value.status_code == 400
    assert "webcam" in e.value.detail.lower() or "connection" in e.value.detail.lower()


# =============================================================================
# Async email (aiosmtplib)
# =============================================================================
# Production code reads ``SMTP_USER`` / ``SMTP_PASS`` from module-level names set
# at import from ``os.getenv``; tests override those attributes on ``main``.


@pytest.mark.asyncio
@patch("main.aiosmtplib.send")
async def test_email_sender_success(mock_send, monkeypatch):
    """
    Happy path: ``aiosmtplib.send`` succeeds; assert kwargs (host, user) and headers.

    The first positional arg to ``send`` is an ``EmailMessage`` (plain + HTML parts).
    """
    monkeypatch.setattr(main, "SMTP_USER", "test@example.com")
    monkeypatch.setattr(main, "SMTP_PASS", "testpass")

    mock_send.return_value = None

    email = "recipient@example.com"
    feedback_content = "This is a test feedback"
    result = await email_sender(email, feedback_content)

    assert result == {"message": "Email sent successfully"}
    mock_send.assert_called_once()
    call_kw = mock_send.call_args[1]
    assert call_kw["hostname"] == main.SMTP_HOST
    assert call_kw["username"] == "test@example.com"
    msg = mock_send.call_args[0][0]
    assert msg["To"] == email
    assert msg["Subject"] == "Your Feedback on your exercise"
    assert feedback_content in str(msg)


@pytest.mark.asyncio
@patch("main.aiosmtplib.send")
async def test_email_sender_failure(mock_send, monkeypatch):
    """Network or server errors are logged; API still returns a soft failure dict (no raise)."""
    monkeypatch.setattr(main, "SMTP_USER", "test@example.com")
    monkeypatch.setattr(main, "SMTP_PASS", "testpass")
    mock_send.side_effect = Exception("SMTP error")

    email = "test@example.com"
    feedback_content = "This is a test feedback"
    result = await email_sender(email, feedback_content)

    assert result == {"message": "Failed to send email"}


@pytest.mark.asyncio
async def test_email_sender_missing_credentials(monkeypatch):
    """Missing creds short-circuit before ``send``; ensures we never open a socket with empty auth."""
    with patch("main.aiosmtplib.send") as mock_send:
        monkeypatch.setattr(main, "SMTP_USER", None)
        monkeypatch.setattr(main, "SMTP_PASS", "x")

        result = await email_sender("test@example.com", "feedback")

        assert result["message"].startswith("Email not sent")
        mock_send.assert_not_called()


# =============================================================================
# User records (async route handler with mocked SQLite)
# =============================================================================


@pytest.mark.asyncio
async def test_update_user_records_success(mocker):
    """
    ``update_user_records`` SELECTs, mutates JSON arrays, then UPDATEs.

    Two different cursor mocks avoid reusing the same ``fetchone`` return for
    both statements (closer to real sqlite3 behavior).
    """
    conn_mock = mocker.MagicMock()
    mocker.patch("main.get_db_connection", return_value=conn_mock)

    email = "test@example.com"
    user_row = {"email": email, "squat": "[]", "bench_press": "[]", "deadlift": "[]"}
    select_cursor = mocker.MagicMock()
    select_cursor.fetchone.return_value = user_row
    update_cursor = mocker.MagicMock()
    conn_mock.execute.side_effect = [select_cursor, update_cursor]

    new_squat = 100.0
    new_bench_press = 150.0
    new_deadlift = 200.0

    response = await update_user_records(
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
