# 🐾 AI Animal Species Predictor

An AI-powered web app that predicts the species of an animal from an uploaded image using Convolutional Neural Networks (CNN).



---

## 📌 Features

- Upload an image of an animal
- Get top predicted species with confidence scores
- Built using TensorFlow/Keras and Flask
- Simple web interface (HTML + CSS)

---

## 🧠 How It Works

1. A Convolutional Neural Network (CNN) model is trained on the [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10).
2. The model learns to classify 10 types of animals (like dog, cat, spider, elephant, etc.).
3. You upload an image on the web interface.
4. The backend processes it, feeds it into the trained model, and returns predictions.

---

## 🔧 Tech Stack

| Area         | Tech Used               |
|--------------|--------------------------|
| Language     | Python                   |
| Framework    | Flask                    |
| Deep Learning| TensorFlow / Keras       |
| Frontend     | HTML + CSS               |

---

## 📂 Folder Structure

project/
├── model/ # Saved model + class_indices.json
├── static/ # CSS + uploaded images
├── templates/ # HTML files
├── dataset/ # ⚠️ Not uploaded (see below)
├── app.py # Main Flask server
├── predict.py # Prediction logic
├── train.py # Model training script
├── requirements.txt # Required libraries
└── README.md # This file

yaml
Copy
Edit

---

## 📦 Dataset

This project uses the **Animals-10 Dataset**.

🔗 Download link: [Kaggle - Animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

> **Note:** The dataset is **not included** in this repository due to large size.

### 📥 Setup Instructions:

1. Download the dataset from the link above.
2. Unzip it and rename the main folder to `dataset/`.
3. Place it in the root project folder.

---

## ▶️ Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
2. Train the model (Optional if you use the pre-trained model)
bash
Copy
Edit
python train.py
3. Start the Flask app
bash
Copy
Edit
python app.py
Then open your browser at: http://127.0.0.1:5000



markdown
Copy
Edit
1. Spider – 92.5%
2. Butterfly – 3.2%
3. Dog – 1.4%
📄 License
MIT License – Use freely with attribution.

✍️ Author
Made by Vansh Salgotra
GitHub Profile

yaml
Copy
Edit
