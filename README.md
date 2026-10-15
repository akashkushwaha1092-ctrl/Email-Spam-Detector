# 📧 Email Spam Detector

An intelligent, machine-learning-powered Email Spam Detection system built with **Python**, **Scikit-Learn**, and **Streamlit**. 

This application classifies emails and text messages into **Spam** or **Ham (Not Spam)** using Natural Language Processing (NLP) and a Support Vector Machine (`LinearSVC`). It features both an interactive web dashboard and a command-line interface (CLI), along with an active feedback loop that enables on-the-fly model retraining.

---

## 🌟 Features

- **High-Accuracy ML Pipeline**:
  - **Feature Extraction**: TF-IDF Vectorizer with unigrams & bigrams (`ngram_range=(1, 2)`), English stop-words removal, and up to 10,000 features.
  - **Classification**: `LinearSVC` (Linear Support Vector Classifier) for fast and robust text classification.
- **Interactive Web App ([app.py](file:///c:/Users/ak933/Desktop/akash/Email-Spam-Detector/app.py))**:
  - **Dataset Management**: Upload custom CSV datasets and preview dataset statistics (total emails, spam count, ham count).
  - **Data Preprocessing**: Automatic missing value handling, type casting, deduplication, and label cleaning.
  - **Model Training & Evaluation**: Train on demand and view accuracy metrics with an 80/20 train-test split.
  - **Real-Time Classification**: Paste any email or message to instantly determine if it is spam or safe.
  - **Feedback & Continuous Learning**: Submit correction feedback on predictions to update the dataset and retrain the classifier in real-time.
- **Command-Line Interface ([cli.py](file:///c:/Users/ak933/Desktop/akash/Email-Spam-Detector/cli.py))**:
  - Full terminal-driven workflow with detailed metrics (Accuracy, Precision, Recall, F1-Score).
  - Interactive prompt for rapid one-off email testing and retraining feedback.

---

## 📁 Project Structure

```text
Email-Spam-Detector/
├── app.py              # Streamlit web application
├── cli.py              # Terminal-based interactive classifier
├── emails_dataset.csv  # Dataset containing email text and labels
└── README.md           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/akashkushwaha1092-ctrl/Email-Spam-Detector.git
cd Email-Spam-Detector
```

### 3. Install Dependencies
Install the required Python packages:

```bash
pip install streamlit pandas scikit-learn
```

---

## 🚀 How to Run

### Option 1: Streamlit Web Dashboard (Recommended)

Launch the interactive web UI:

```bash
streamlit run app.py
```

Once started, open your browser at `http://localhost:8501`.

1. **Upload Dataset** *(Optional)*: Upload an updated CSV or proceed with the included [emails_dataset.csv](file:///c:/Users/ak933/Desktop/akash/Email-Spam-Detector/emails_dataset.csv).
2. **Train Model**: Click **"🚀 Train Spam Detection Model"** to fit the classifier on the dataset.
3. **Check Email**: Enter or paste any email text and click **"🔍 Detect Spam"**.
4. **Provide Feedback**: If a prediction was inaccurate, choose the correct label and click **"Submit Feedback"**, then retrain the model.

---

### Option 2: Command-Line Interface (CLI)

Run the CLI tool directly in your terminal:

```bash
python cli.py
```

- Displays dataset statistics and cleaning logs.
- Trains the model and prints evaluation metrics (Accuracy, Precision, Recall, F1).
- Prompts for email input to classify in real time (`type 'exit' to quit`).

---

## 📊 Dataset Format

The dataset ([emails_dataset.csv](file:///c:/Users/ak933/Desktop/akash/Email-Spam-Detector/emails_dataset.csv)) should be a CSV file with at least the following two columns:

| Column | Description | Format / Values |
| :--- | :--- | :--- |
| `text` | The body/content of the email or message | String |
| `spam` | Label indicating whether the email is spam | `1` (Spam) or `0` (Not Spam / Ham) |

---

## 🛠️ Technology Stack

- **Language**: Python 3
- **Web Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/) (`LinearSVC`, `TfidfVectorizer`, `Pipeline`)
- **Data Manipulation**: [pandas](https://pandas.pydata.org/)
