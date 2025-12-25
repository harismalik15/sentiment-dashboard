import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load data from folders
def load_data(data_dir):
    texts, labels = [], []
    for label, folder in [(1, "pos"), (0, "neg")]:
        path = os.path.join(data_dir, folder)
        for file in os.listdir(path):
            with open(os.path.join(path, file), encoding="utf-8") as f:
                texts.append(f.read())
                labels.append(label)
    return texts, labels


print("Loading data...")
texts, labels = load_data("data/train")   # ✅ correct place to call it

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/tfidf.pkl")

print("Model saved in models/ folder")
