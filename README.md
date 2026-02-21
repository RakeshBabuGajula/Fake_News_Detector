# 📰 Fake News Detection using LSTM (Deep Learning)

🚀 An end-to-end AI-powered Fake News Detection system built using LSTM Neural Networks and deployed as a live web application on Hugging Face.
<img width="1536" height="1024" alt="Fake_news_detector_thumbnail" src="https://github.com/user-attachments/assets/fac18391-50a3-4163-95b5-645d96caff7d" />


---

## 📌 Project Overview

With the rapid growth of digital media, misinformation and fake news spread quickly across social platforms.  
This project aims to build a deep learning-based system that classifies news articles as **REAL or FAKE** using Natural Language Processing (NLP).

The system is trained on labeled news data and deployed as an interactive web application.

---

## 🎯 Features

- ✅ Text preprocessing and cleaning pipeline
- ✅ LSTM-based Deep Learning model
- ✅ ~95% validation accuracy
- ✅ Real-time prediction via Streamlit Web App
- ✅ Deployed on Hugging Face Spaces
- ✅ End-to-end ML workflow implementation

---

## 🧠 Model Architecture

The deep learning architecture consists of:

- **Embedding Layer** – Converts words into dense vectors
- **LSTM Layer (64 units)** – Captures sequential dependencies
- **Dropout Layer (0.3)** – Prevents overfitting
- **Dense Output Layer (Sigmoid)** – Binary classification (Real/Fake)

Loss Function: `Binary Crossentropy`  
Optimizer: `Adam`

---

## 🔄 Workflow

1. Data Collection (Kaggle Fake News Dataset)
2. Data Preprocessing
   - Remove special characters
   - Lowercasing
   - Tokenization
   - Padding sequences
3. Model Training using TensorFlow/Keras
4. Model Saving (.keras format)
5. Streamlit App Development
6. Deployment on Hugging Face Spaces

---

## 📊 Results

- 📈 Training Accuracy: ~96%
- 📈 Validation Accuracy: ~95%
- ⚡ Real-time prediction performance
- 🌍 Successfully deployed online

---

## 🌐 Live Demo

🔗 **Hugging Face Deployment Link:**  

https://huggingface.co/spaces/RakeshBabuGajula/Fake-News-Detector
<img width="1917" height="1017" alt="Screenshot 2026-02-21 172234" src="https://github.com/user-attachments/assets/e8323eb2-ffb7-424f-acd1-14b50ac1caa5" />
<img width="1919" height="1017" alt="Screenshot 2026-02-21 172943" src="https://github.com/user-attachments/assets/a9120d3a-da15-4bed-8a89-ad36f11e5d85" />
<img width="1919" height="1019" alt="Screenshot 2026-02-21 172800" src="https://github.com/user-attachments/assets/49c1b411-997e-4c33-9f2e-c6c1d8d204a1" />
<img width="1918" height="1017" alt="Screenshot 2026-02-21 173009" src="https://github.com/user-attachments/assets/bd78d7d5-346f-41ab-97c4-10fbe591c742" />


---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas & NumPy
- Streamlit
- Hugging Face Spaces

---

## 📁 Project Structure


fake-news-detector/
│
├── app.py
├── fake_news_model.keras
├── tokenizer.pkl
├── requirements.txt
├── README.md


---

## 🚀 Installation (Local Setup)

Clone the repository:


git clone https://github.com/your-username/fake-news-detector.git

cd fake-news-detector


Install dependencies:


pip install -r requirements.txt


Run the application:


streamlit run app.py


---

## 🔮 Future Improvements

- Upgrade to BERT / Transformer model
- Add multilingual support
- Add explainable AI (highlight suspicious words)
- URL-based news prediction
- Mobile app version

---

## 📚 References

- TensorFlow Documentation
- Streamlit Documentation
- Hugging Face Spaces
- Kaggle Fake News Dataset

---

## 👨‍💻 Author

**Rakesh Babu Gajula**  
B.Tech - Computer Science Engineering  
AI & Machine Learning Enthusiast  

---

⭐ If you like this project, give it a star!
