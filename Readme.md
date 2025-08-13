This project detects whether a news article is **Fake** or **Real** using Machine Learning and Natural Language Processing (NLP).

Fake News Detection (Streamlit + Scikit-learn)

Detect whether a news article is Real or Fake using a TF-IDF + Logistic Regression pipeline.
Includes a Streamlit UI, explainability, test suite, and Dockerized deployment.

====================================

FEATURES
====================================

Text preprocessing + TF-IDF features

Logistic Regression classifier

Probability output + credibility score

Token-level explanation: words pushing toward Real vs Fake

Streamlit UI (app.py)

Docker image for one-click run anywhere

Pytest tests for artifacts, predictions, and edge cases

Training/validation accuracy & loss graphs included in the report

====================================
2. PROJECT STRUCTURE
app.py -> Streamlit app
requirements.txt -> Python dependencies
vectorizer.joblib -> Saved TF-IDF vectorizer
lr_model.joblib -> Saved Logistic Regression model
Dockerfile -> Container build file
tests/ -> Test cases (pytest)

Make sure vectorizer.joblib and lr_model.joblib are in the project root.

====================================
3. REQUIREMENTS (Local Development)
Python 3.11

pip, virtualenv (or venv)

(Optional) Git, VS Code

(For containers) Docker Desktop (Windows/macOS) or Docker Engine (Linux)

====================================
4. RUN LOCALLY (Without Docker)
Create & activate a virtual environment
python -m venv venv
On Windows PowerShell: .\venv\Scripts\Activate.ps1
On macOS/Linux: source venv/bin/activate

Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

Run the app
streamlit run app.py --server.port 8501

Open: http://localhost:8501

====================================
5. RUN TESTS
pip install pytest
pytest -q

====================================
6. RUN WITH DOCKER
Build the image
docker build -t fake-news-app:latest .

Run the container
docker run --rm -p 8501:8501 fake-news-app:latest

Open: http://localhost:8501

====================================
7. TROUBLESHOOTING
Docker errors about connection or missing pipe: make sure Docker Desktop is running and WSL backend is active.

Input/output error during build: free disk space and clear Docker cache.

NumPy / scikit-learn errors: activate correct venv and reinstall dependencies.

Missing model artifacts: ensure vectorizer.joblib and lr_model.joblib are in the root folder.

====================================
8. HOW IT WORKS
Preprocess text (lowercase, remove punctuation/extra spaces, remove stopwords)

TF-IDF transforms text to sparse features

Logistic Regression outputs class probabilities

Credibility score shown with token-level explanation

====================================
9. SUBMISSION CHECKLIST
app.py runs locally and in Docker

vectorizer.joblib & lr_model.joblib included

requirements.txt installs cleanly

Tests pass with pytest

Screenshots and graphs added to report

README.txt updated with instructions

