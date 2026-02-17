# Powerlifting Trainer Assistant API 💪🤖

A FastAPI backend that manages lifter profiles (BMI + PR history) and provides **AI-based coaching feedback** using OpenAI.  
It also supports **exercise form analysis** using **MediaPipe + OpenCV** for:
- **Uploaded video analysis (MP4)**
- **Real-time webcam analysis (local only)**

This project was built as a **final project** for the course **Software Development Using AI**.

**Created by:** Dor Haboosha, Itay Golan, Moran Herzlinger  
**Project video:** https://youtu.be/xNZTbHj9JV0

---

## ✅ Features

### 👤 Users & Records
- Register a lifter profile (email, height, weight, name)
- Auto-calculate BMI
- Store PR history for:
  - Deadlift
  - Squat
  - Bench press
- Fetch profile by email
- Update PR records

### 🎥 Video Analysis
- Upload an `.mp4` video → analyze form using MediaPipe + OpenCV → get AI feedback
- Real-time webcam mode (counts “good reps” + AI feedback)

---

## ⚠️ Important Limitations

### Real-time webcam mode
✅ Works when running **locally on your machine** (because it requires direct access to the webcam).

❌ Does **NOT** work in **Docker** or any cloud deployment (no webcam access).  
In Docker/cloud, the API returns a friendly message telling the user to upload a video instead.

### Email sending (SMTP)
✅ **Works locally and in Docker** (when the environment/network allows SMTP traffic and credentials are configured).

⚠️ Some cloud providers can block SMTP ports (or require extra setup), so email may fail in hosted deployments.

---

## 🧩 API Documentation (Swagger)

When the server is running, open:

- **Swagger UI:** `http://localhost:8000/docs`
- **Health Check:** `http://localhost:8000/health_check`

Root path redirects to `/docs`.

---

## 🛠️ Run Locally (Recommended)

### 1) Create and activate virtual environment
```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Create `.env`
Create a `.env` file in the project root (based on `.env.example`) and set your keys.

### 4) Run the server
```bash
python main.py
```

Open: `http://localhost:8000/docs`

✅ In local mode:
- Uploaded video analysis works
- Real-time webcam analysis works
- SMTP email sending works (if configured)

---

## 🐳 Run with Docker

### 1) Build the image
```bash
docker build -t powerlifting-api .
```

### 2) Run the container
```bash
docker run --rm -it \
  --name powerlifting-api \
  --env-file .env \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  powerlifting-api
```

Open: `http://localhost:8000/docs`

✅ In Docker mode:
- Uploaded video analysis works
- API + DB work (via mounted volume)
- SMTP email sending works **if Docker/network allows SMTP** and credentials are valid

❌ Real-time webcam mode does not work (expected)  
You will get the friendly message telling you to upload a video instead.

> **Windows PowerShell volume tip:**  
> Replace `-v $(pwd)/data:/data` with `-v ${PWD}\data:/data`

---

## 🧪 Run Unit Tests
```bash
pytest -q
```

---

## ☁️ Azure Pipeline (Course Requirement)

As part of the course instructions, the project was intended to be connected to Azure for CI/CD deployment.

This repository includes an Azure pipeline configuration file:
- `azure-pipelines-1.yml`

✅ The pipeline file exists and can be used to connect the project to Azure DevOps and build/push a Docker image.

⚠️ The Azure deployment is **not currently connected/active** due to technical/account limitations.

---

## 🔐 Security Notes
- ✅ Your `.env` file should stay **local only** and is already excluded via `.gitignore`.
- ✅ Use `.env.example` as a template for required environment variables.

---

## 📦 Tech Stack
- **FastAPI** (Python)
- **SQLite**
- **OpenCV**
- **MediaPipe**
- **OpenAI API**
- **Docker**
- **Pytest**
