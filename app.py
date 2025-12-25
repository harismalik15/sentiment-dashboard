import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf.pkl")

# Page configuration
st.set_page_config(page_title="Sentiment Analysis Dashboard", layout="wide")

# Title
st.title("Sentiment Analysis Dashboard")
st.write("Predict sentiment for single text input or upload a CSV file of texts.")

# Tabs for single text or batch CSV
tab1, tab2 = st.tabs(["Single Text Prediction", "Batch CSV Prediction"])

# -------------------------
# SINGLE TEXT PREDICTION
# -------------------------
with tab1:
    st.header("Single Text Prediction")
    text = st.text_area("Enter your text here:")

    if st.button("Analyze Text", key="single"):
        if text.strip() == "":
            st.warning("Please enter some text.")
        else:
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            label = "Positive 😊" if pred == 1 else "Negative 😞"
            confidence = max(prob) * 100

            # Layout using columns
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Predicted Sentiment")
                st.write(label)
            with col2:
                st.subheader("Confidence")
                st.write(f"{confidence:.2f}%")

# -------------------------
# BATCH CSV PREDICTION
# -------------------------
with tab2:
    st.header("Batch CSV Prediction")
    st.write("Upload a CSV file with a column named `text` containing your sentences.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        if 'text' not in df.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())

            # Predict sentiment
            X_vec = vectorizer.transform(df['text'])
            preds = model.predict(X_vec)
            probs = model.predict_proba(X_vec)

            df['sentiment'] = ['Positive' if p==1 else 'Negative' for p in preds]
            df['confidence'] = [max(p)*100 for p in probs]

            st.subheader("Predictions")
            st.dataframe(df)

            # Pie chart of sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
            ax.axis('equal')
            st.subheader("Sentiment Distribution")
            st.pyplot(fig)

            # Optionally, allow download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='sentiment_predictions.csv',
                mime='text/csv'
            )
