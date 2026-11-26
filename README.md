# 🎬 Simple RNN – IMDB Movie Review Sentiment Analysis

## 📌 Project Overview

This project uses a Simple Recurrent Neural Network (Simple RNN) to perform sentiment analysis on movie reviews.

The model analyzes the text of a movie review and classifies it into two categories:

- **Positive Sentiment**
- **Negative Sentiment**

An interactive web application is developed using Streamlit, allowing users to enter movie reviews and get sentiment predictions.

## 🎯 Project Objective

The main objective of this project is to understand Natural Language Processing (NLP) and Deep Learning concepts by building a sentiment classification model using Simple RNN.

## 📊 Dataset

- **Dataset:** IMDB Movie Reviews Dataset
- **Source:** TensorFlow Keras Datasets
- **Task:** Binary Sentiment Classification
- **Classes:** Positive and Negative

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Streamlit
- Natural Language Processing (NLP)
- Simple Recurrent Neural Network (Simple RNN)

## 🧠 Model Architecture

The model consists of the following layers:

1. Input Layer
2. Embedding Layer (64 dimensions)
3. Simple RNN Layer (32 units)
4. Dense Layer (Sigmoid Activation)

### Model Configuration

- Vocabulary Size: 10,000
- Maximum Sequence Length: 300
- Batch Size: 64
- Epochs: 5
- Early Stopping: Used during training

## 📁 Project Structure

```text
Simple_RNN/
│
├── app.py
├── embedding.ipynb
├── prediction.ipynb
├── simple_rnn_imdb.h5
├── requirements.txt
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/chandanidevare/Simple-RNN-IMDB-Sentiment-Analysis.git
```

### 2. Navigate to Project Folder

```bash
cd Simple-RNN-IMDB-Sentiment-Analysis
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Virtual Environment (Windows)

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

## 💡 Application Features

- Interactive movie review input
- Positive and negative sentiment prediction
- Prediction score display
- Sample review testing
- User-friendly Streamlit interface

## 📈 Results

The trained Simple RNN model predicts whether a movie review expresses positive or negative sentiment.

Users can enter their own movie reviews and view the model's prediction through the Streamlit web application.

## 👩‍💻 Author

**Chandani Devare**

GitHub: https://github.com/chandanidevare

---

⭐ If you find this project interesting, feel free to explore the repository!
