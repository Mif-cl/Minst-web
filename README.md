# 🧠 MNIST Handwritten Digit Classifier

[![Streamlit App](https://img.shields.io/badge/Launch%20App-Streamlit-brightgreen?logo=streamlit)](https://minst-web.streamlit.app/)

This is a web-based interactive application that lets users draw digits (0–9) and classify them using a trained PyTorch model. The app also lets users provide feedback to improve model performance over time by logging their corrections to a PostgreSQL database.

🔗 **Live App:** [https://minst-web.streamlit.app/](https://minst-web.streamlit.app/)

## 🌐 Deployment Note

While the original goal included deploying to a self-managed server (e.g., Hetzner or other VPS with Docker), this app is currently hosted on **[Streamlit Community Cloud](https://streamlit.io/cloud)** for the following reasons:

- 💸 **Zero-cost hosting** — ideal for public prototypes and demos.
- 🚀 **Quick setup and CI/CD via GitHub integration.**
- 🔄 **Easily portable** — a `Dockerfile` is included to enable migration to any self-managed server or cloud platform in the future.

If needed, this app can be deployed on a private VPS with Docker using the provided configuration. See the Docker section below for instructions.


## ✨ Features

- 🎨 **Interactive Drawing Canvas** — Draw a digit directly in the browser.
- 🔍 **Digit Prediction** — Get the predicted digit and confidence from a trained PyTorch model.
- ✅ **User Feedback Input** — Manually correct predictions for future model improvements.
- 🗃️ **Feedback Logging** — All predictions and user-labeled corrections are stored in a PostgreSQL database.
- 🚀 **Deployed via Docker** — Easily portable and cloud-deployable.

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/) — Web interface
- [PyTorch](https://pytorch.org/) — Model inference
- [PostgreSQL](https://www.postgresql.org/) — Feedback storage
- [Docker](https://www.docker.com/) — Containerized deployment

---

## 📦 Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/minst-web.git
cd minst-web
```

### 2. Create a `.streamlit/secrets.toml` File
```toml
[database]
host = "your-db-host.supabase.co"
port = 5432
database = "postgres"
user = "postgres"
password = "your-db-password"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run scr/app.py
```

---

## 🐳 Docker Deployment

### Build the Docker image
```bash
docker build -t mnist-web-app .
```

### Run the container
```bash
docker run -p 8501:8501 mnist-web-app
```

Then visit `http://localhost:8501` or your server's IP.

---

## 📁 Project Structure

```
Minst-web/
├── checkpoint.pt
├── Dockerfile
├── requirements.txt
├── scr/
│   ├── app.py
│   ├── ImageClassifier.py
│   └── Mnist_infer.py
└── .streamlit/
    └── secrets.toml
```

---