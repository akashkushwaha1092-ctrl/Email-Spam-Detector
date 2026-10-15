import streamlit as st
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATASET_FILE = "emails_dataset.csv"
# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

def load_dataset():
    if os.path.exists(DATASET_FILE):
        df = pd.read_csv(DATASET_FILE)
        return df

    return pd.DataFrame(columns=["text", "spam"])


# --------------------------------------------------
# Save Dataset
# --------------------------------------------------

def save_dataset(df):
    df.to_csv(DATASET_FILE, index=False)


# --------------------------------------------------
# Clean Dataset
# --------------------------------------------------

def clean_dataset(df):

    # Keep required columns
    df = df[["text", "spam"]].copy()

    # Remove empty rows
    df = df.dropna()

    # Convert text to string
    df["text"] = df["text"].astype(str)

    # Convert spam column to numeric
    df["spam"] = pd.to_numeric(
        df["spam"],
        errors="coerce"
    )

    # Remove invalid labels
    df = df.dropna(subset=["spam"])

    # Keep only 0 and 1
    df = df[df["spam"].isin([0, 1])]

    # Convert numeric labels to spam/ham
    df["label"] = df["spam"].map({
        0: "ham",
        1: "spam"
    })

    # Remove duplicate emails
    df = df.drop_duplicates(
        subset=["text"],
        keep="last"
    )

    return df

# --------------------------------------------------
# Train Model
# --------------------------------------------------

def train_model(df):

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                max_features=10000,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LinearSVC()
        )
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="Spam Mail Detector",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Spam Mail Detection System")

st.write(
    "Upload your dataset, train the model, and detect whether "
    "an email/message is Spam or Not Spam."
)


# --------------------------------------------------
# Dataset Upload
# --------------------------------------------------

st.header("1️⃣ Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is not None:

    uploaded_df = pd.read_csv(uploaded_file)

    st.write("Uploaded Dataset Preview:")
    st.dataframe(uploaded_df.head())

    if "text" not in uploaded_df.columns or "spam" not in uploaded_df.columns:

        st.error(
            "Dataset must contain these columns: text and spam"
        )

    else:

        uploaded_df = clean_dataset(uploaded_df)

        if len(uploaded_df) == 0:

            st.error(
                "No valid data found. Make sure labels are spam and ham."
            )

        else:

            save_dataset(uploaded_df)

            st.success(
                f"Dataset uploaded successfully! "
                f"{len(uploaded_df)} records saved."
            )


# --------------------------------------------------
# Load Current Dataset
# --------------------------------------------------

df = load_dataset()

df = clean_dataset(df)


# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.header("2️⃣ Dataset Information")

if len(df) > 0:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Emails",
            len(df)
        )

    with col2:
        st.metric(
            "Spam Emails",
            len(df[df["label"] == "spam"])
        )

    with col3:
        st.metric(
            "Normal Emails",
            len(df[df["label"] == "ham"])
        )

    with st.expander("View Dataset"):
        st.dataframe(df)

else:

    st.warning(
        "Please upload a dataset first."
    )


# --------------------------------------------------
# Train Model
# --------------------------------------------------

model = None
accuracy = None

if len(df) >= 10:

    st.header("3️⃣ Train Model")

    if st.button("🚀 Train Spam Detection Model"):

        try:

            model, accuracy = train_model(df)

            st.session_state["model"] = model
            st.session_state["accuracy"] = accuracy

            st.success("Model trained successfully!")

        except Exception as e:

            st.error(
                f"Model training failed: {e}"
            )


# --------------------------------------------------
# Display Accuracy
# --------------------------------------------------

if "accuracy" in st.session_state:

    st.info(
        f"Model Accuracy: "
        f"{st.session_state['accuracy'] * 100:.2f}%"
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

st.header("4️⃣ Check Email")

email_text = st.text_area(
    "Enter email/message",
    height=200,
    placeholder="Paste your email or message here..."
)


if st.button("🔍 Detect Spam"):

    if "model" not in st.session_state:

        st.warning(
            "Please train the model first."
        )

    elif not email_text.strip():

        st.warning(
            "Please enter an email/message."
        )

    else:

        model = st.session_state["model"]

        prediction = model.predict(
            [email_text]
        )[0]

        st.session_state["last_email"] = email_text
        st.session_state["last_prediction"] = prediction

        if prediction == "spam":

            st.error(
                "🚨 This email is SPAM"
            )

        else:

            st.success(
                "✅ This email is NOT SPAM"
            )


# --------------------------------------------------
# Feedback System
# --------------------------------------------------

if "last_prediction" in st.session_state:

    st.header("5️⃣ Feedback")

    st.write(
        f"Model prediction: **{st.session_state['last_prediction'].upper()}**"
    )

    st.write(
        "Was this prediction correct?"
    )

    feedback = st.radio(
        "Select feedback",
        [
            "Correct",
            "Wrong - It is actually Spam",
            "Wrong - It is actually Not Spam"
        ],
        key="feedback"
    )

    if st.button("Submit Feedback"):

        email = st.session_state["last_email"]
        prediction = st.session_state["last_prediction"]

        # Determine correct label
        if feedback == "Correct":

            correct_label = prediction

        elif feedback == "Wrong - It is actually Spam":

            correct_label = "spam"

        else:

            correct_label = "ham"

        # Add feedback to dataset
        new_row = pd.DataFrame({
            "text": [email],
            "spam": [1 if correct_label == "spam" else 0]
        })

        df = pd.concat(
            [df, new_row],
            ignore_index=True
        )

        # Remove duplicate emails
        df = df.drop_duplicates(
            subset=["text"],
            keep="last"
        )

        save_dataset(df[["text", "spam"]])

        st.success(
            f"Feedback saved! The email was added as '{correct_label}'."
        )

        st.info(
            "Retrain the model to make it learn from the new feedback."
        )


# --------------------------------------------------
# Retrain Model
# --------------------------------------------------

st.header("6️⃣ Retrain With Updated Dataset")

if st.button("🔄 Retrain Model"):

    df = load_dataset()
    df = clean_dataset(df)

    if len(df) >= 10:

        try:

            model, accuracy = train_model(df)

            st.session_state["model"] = model
            st.session_state["accuracy"] = accuracy

            st.success(
                "Model retrained using the updated dataset!"
            )

            st.info(
                f"New Accuracy: {accuracy * 100:.2f}%"
            )

        except Exception as e:

            st.error(
                f"Retraining failed: {e}"
            )

    else:

        st.warning(
            "You need at least 10 records to train the model."
        )