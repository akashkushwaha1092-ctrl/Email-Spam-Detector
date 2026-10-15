import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
DATASET = "emails_dataset.csv"
# ==============================
# LOAD AND CLEAN DATA
# ==============================
def load_data():
    df = pd.read_csv(DATASET)
    print("\nOriginal columns:")
    print(df.columns.tolist())
    # Keep required columns
    df = df[["text", "spam"]]
    # Remove empty data
    df = df.dropna()
    # Convert text to string
    df["text"] = df["text"].astype(str)
    # Convert spam column to integer
    df["spam"] = pd.to_numeric(
        df["spam"],
        errors="coerce"
    )
    # Remove invalid labels
    df = df.dropna(
        subset=["spam"]
    )
    # Only 0 and 1
    df = df[
        df["spam"].isin([0, 1])
    ]
    # Remove duplicate emails
    df = df.drop_duplicates(
        subset="text"
    )
    print("\nClean Dataset")
    print("------------------------")
    print("Total emails:", len(df))
    print("Normal emails:", sum(df["spam"] == 0))
    print("Spam emails:", sum(df["spam"] == 1))
    return df
# ==============================
# TRAIN MODEL
# ==============================
def train_model(df):
    X = df["text"]
    y = df["spam"]
    # 80% training
    # 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    # TF-IDF + LinearSVC
    model = Pipeline([

        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=20000
            )
        ),

        (
            "classifier",
            LinearSVC()
        )
    ])
    print("\nTraining model...")
    model.fit(
        X_train,
        y_train
    )
    print("Training complete!")
    # Test model
    prediction = model.predict(
        X_test
    )
    # Metrics
    accuracy = accuracy_score(
        y_test,
        prediction
    )
    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )
    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")
    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )
    print(
        f"Precision : {precision * 100:.2f}%"
    )
    print(
        f"Recall    : {recall * 100:.2f}%"
    )
    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )
    return model
# ==============================
# ADD USER FEEDBACK
# ==============================
def add_feedback(email, correct_label):
    df = pd.read_csv(DATASET)
    # Add new email
    new_row = pd.DataFrame({
        "text": [email],
        "spam": [correct_label]
    })
    df = pd.concat(
        [df, new_row],
        ignore_index=True
    )
    # Remove duplicate email
    df = df.drop_duplicates(
        subset="text",
        keep="last"
    )
    # Save dataset
    df.to_csv(
        DATASET,
        index=False
    )
    print("\n✅ Feedback added to dataset.")
# ==============================
# MAIN PROGRAM
# ==============================
print("==============================")
print("      SPAM MAIL DETECTOR")
print("==============================")
# Load dataset
df = load_data()
# Train model
model = train_model(df)
# ==============================
# TEST EMAIL
# ==============================
while True:
    print("\n------------------------------")
    email = input(
        "\nEnter email or type 'exit': "
    )
    if email.lower() == "exit":
        print("\nProgram closed.")
        break
    if email.strip() == "":
        print("Please enter an email.")
        continue
    # Predict
    prediction = model.predict(
        [email]
    )[0]
    # Show prediction
    if prediction == 1:
        print("\n🚨 SPAM EMAIL")
    else:
        print("\n✅ NOT SPAM")
    # ==============================
    # FEEDBACK
    # ==============================
    feedback = input(
        "\nIs this prediction correct? (y/n): "
    )
    if feedback.lower() == "y":
        print("✅ Feedback recorded.")
    elif feedback.lower() == "n":
        correct = input(
            "Enter correct label "
            "(1 = spam, 0 = not spam): "
        )
        if correct not in ["0", "1"]:
            print(
                "❌ Invalid label."
            )
            continue
        correct_label = int(correct)
        # Add feedback
        add_feedback(
            email,
            correct_label
        )
        # Reload dataset
        df = load_data()
        # Retrain model
        model = train_model(df)
        print(
            "\n🔄 Model retrained using your feedback!"
        )