#  Fake News Detection Project

This project detects whether a news article is **Fake** or **Real** using Machine Learning and Natural Language Processing (NLP).

---

##  Features
- Preprocessing of text data using TF-IDF
- Logistic Regression classifier
- Streamlit-based user interface
- Saved model and vectorizer using joblib

---

##  Project Structure
├── app.py # Streamlit UI
├── app.ipynb # Model training script
├── vectorizer.joblib # Saved TF-IDF vectorizer
├── lr_model.joblib # Trained Logistic Regression model
├── README.md # Project documentation

How It Works
The Fake News Detection system uses Natural Language Processing (NLP) and Machine Learning to analyze news articles and determine whether they are fake or real.

Here’s how the pipeline works:
 
 User Input (via Streamlit)
The user enters a news article or snippet in a text box on the Streamlit web interface.

Text Preprocessing (TF-IDF Vectorization):
The entered text is transformed into numerical features using a pre-trained TF-IDF (Term Frequency–Inverse Document Frequency) vectorizer.

 Prediction (Using Trained Model):
The numerical features are passed to a trained Logistic Regression model that has learned from thousands of labeled fake and real news articles.

The model outputs a prediction:

1 → Real News
0 → Fake News

Output Display (Streamlit UI)

Based on the model’s prediction, the app shows:
“The News is Real” (if prediction is 1)
“The News is Fake” (if prediction is 0)

 Models and Vectorizer Saved with Joblib

The trained model and vectorizer are saved as .joblib files and loaded instantly when the app runs.

Run the App:

streamlit run app.py


Results:
Accuracy: 99%


Dataset
Used the Fake News Dataset from Kaggle.



