"""
Auto-generated Python script converted from randomforestgraph.ipynb
Contains code cells concatenated in order. You may need to adjust paths
or inline magic commands (e.g., %matplotlib inline) before running.
"""


# ---- Code cell 1 ----

# Email Spam Detection with Random Forest in VSCode
# Made by: Random Forest Model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --------------------
# 1. Load dataset
# --------------------
data = pd.read_csv("spam.csv", encoding="latin-1")
data = data[['v1', 'v2']]  # Keep only label & text columns
data.columns = ['label', 'message']

# --------------------
# 2. Preprocessing
# --------------------
data['label'] = data['label'].map({'ham': 0, 'spam': 1})  # Binary labels
data['message'] = data['message'].str.lower().str.strip()  # Normalize text

# --------------------
# 3. Dataset distribution graph
# --------------------
plt.figure(figsize=(5,4))
sns.countplot(x='label', data=data, palette='coolwarm')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.title("📊 Distribution of Messages")
plt.show()

# --------------------
# 4. Train-test split
# --------------------
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------
# 5. TF-IDF Vectorization
# --------------------
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# --------------------
# 6. Model Training (Random Forest)
# --------------------
model = RandomForestClassifier(
    n_estimators=100,  # Number of trees
    max_depth=20,      # Max depth of each tree
    random_state=42
)
model.fit(X_train_tfidf, y_train)

# --------------------
# 7. Evaluation
# --------------------
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)

print("📊 Model Evaluation:")
print("Accuracy:", acc)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------
# 8. Confusion Matrix Heatmap
# --------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("🧮 Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# --------------------
# 9. Accuracy Bar Graph
# --------------------
plt.figure(figsize=(4,3))
plt.bar(["Accuracy"], [acc], color='green')
plt.ylim(0, 1)
plt.title("✅ Model Accuracy")
plt.show()

# --------------------
# 10. Prediction Function
# --------------------
def predict_spam(email):
    email_tfidf = tfidf.transform([email.lower().strip()])
    prediction = model.predict(email_tfidf)
    return "🚨 SPAM!" if prediction == 1 else "✅ NOT SPAM."

# --------------------
# 11. Example Predictions
# --------------------
print("\n🔍 Test Examples:")
print("1. 'Free prize! Click now!' →", predict_spam("Free prize! Click now!"))
print("2. 'Hey, how are you?' →", predict_spam("Hey, how are you?"))


