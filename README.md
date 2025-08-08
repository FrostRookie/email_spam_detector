# 📧 Email Spam Detection with Machine Learning

This project uses Python and machine learning techniques to build a spam email classifier using the SMS Spam Collection Dataset. It applies natural language processing (NLP) to distinguish between spam and ham (non-spam) messages.

---

## 📁 Dataset

- **Source**: SMS Spam Collection Dataset (from UCI/Kaggle)
- **Columns**:
  - `label`: "ham" or "spam"
  - `message`: the text message content

---

## 🧠 Machine Learning Model

- **Text Vectorization**: TF-IDF (`TfidfVectorizer`)
- **Classifier**: Multinomial Naive Bayes (ideal for text classification)

---

## 📊 Results

| Label | Count |
|-------|-------|
| Ham   | 4825  |
| Spam  | 747   |

- **Accuracy**: ~98%
- **Evaluation Metrics**: Confusion Matrix, Classification Report

---

## 📉 Visualizations

- Label distribution bar chart
- Confusion matrix heatmap
- Sample prediction outputs

---

## 📦 Requirements

- Python 3.x
- pandas
- scikit-learn
- matplotlib
- seaborn

---

## 🚀 How to Run

1. Upload the dataset (`spam.csv`) to your notebook or environment.
2. Run the Jupyter Notebook: `Email_Spam_Detection_Kaggle.ipynb`
3. View model performance, graphs, and try predictions on custom messages.

---

## 📌 Sample Predictions
Message: "Congratulations! You've won a free ticket to Bahamas. Call now!"
Prediction: Spam

Message: "Hey, are we still meeting for lunch today?"
Prediction: Ham

Made by Shubh Patel

