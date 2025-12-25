# Sentiment Analysis Dashboard

A **web-based Sentiment Analysis Dashboard** built using Python, scikit-learn, and Streamlit.  
This dashboard allows users to **predict sentiment** (Positive/Negative) from text in **real-time** or in **batch via CSV upload**, with confidence scores and visualizations.

---

## Features

- **Single Text Prediction**: Enter any text and get sentiment prediction instantly.  
- **Batch CSV Prediction**: Upload a CSV file with a `text` column and get predictions for multiple sentences.  
- **Confidence Scores**: Shows how confident the model is in its prediction.  
- **Visualizations**: Pie chart displaying sentiment distribution in batch predictions.  
- **Download Results**: Save batch predictions as a CSV file.  
- **Clean and Interactive UI**: Built with Streamlit, responsive and user-friendly.

---

## Demo

**Single Text Prediction:**  


**Batch CSV Prediction:**  

| text                                   | sentiment | confidence |
|----------------------------------------|-----------|------------|
| "Worst movie ever."                     | Negative  | 90.2%      |
| "Absolutely loved it!"                  | Positive  | 94.1%      |

Pie chart shows sentiment distribution for all texts in the CSV.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/harismalik15/sentiment-dashboard.git
cd sentiment-dashboard
```

2. **Clone the repository**
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

3. **Install dependencies**
pip install -r requirements.txt

4. **Train the model**
python train.py
Uses the included sample dataset in data/ folder. You can also provide your own dataset.

5. **Run the Streamlit Dashboard**
streamlit run app.py

6. **📂 Project Structure**
sentiment_dashboard/
- **data**: Sample dataset (optional: you can add your own)
- **images**: Screenshots for README
- **model**: Trained model and TF-IDF vectorizer
- **app.py**: Streamlit dashboard
- **train.py**: Model training script
- **requirements.txt**: Dependencies
- **gitignore**: Ignored files
- **README.md**: Project documentation

---

## 🛠 Technologies Used

- **Python 3.11**: Core programming language.  
- **scikit-learn**: TF-IDF vectorization & Logistic Regression.  
- **Streamlit**: Interactive web dashboard.  
- **Pandas**: Data handling and manipulation.  
- **Matplotlib**: Visualizations for sentiment distribution.  
- **Clean and Interactive UI**: Built with Streamlit, responsive and user-friendly.
---
## 📈 How It Works

- **Train the Model**: train.py reads the dataset, converts text to numeric vectors using TF-IDF, and trains a Logistic Regression model.
- **Run Dashboard**: app.py loads the trained model and allows users to:
- Enter single text for prediction
- Upload CSV for batch predictions
- Display Results – Predictions, confidence scores, and visualizations are displayed in a clean UI.
- Download Batch results can be downloaded as CSV for further analysis.
---

## 🚀 Future Improvements

- Add **neutral sentiment** class (Positive / Negative / Neutral)
- Upgrade Logistic Regression to BERT / DistilBERT for higher accuracy
- Deploy dashboard on Streamlit Cloud / Heroku
- Add interactive bar charts for confidence scores and trends over time
---

## 🔗 Links
- https://docs.streamlit.io/
- https://scikit-learn.org/
---
  
## 📜 License
- This project is MIT Licensed. Feel free to use for learning, portfolio, or demo purposes.




