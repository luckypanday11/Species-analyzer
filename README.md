# 🌸 Iris Species Analyzer

An interactive desktop application that predicts the species of an Iris flower using a **K-Nearest Neighbors (KNN)** machine
learning model. The application provides a modern graphical interface built with **CustomTkinter**, allowing users to enter
flower measurements and receive real-time predictions along with confidence scores.

---

## 📌 Overview

The Iris Species Analyzer demonstrates a complete machine learning workflow, from data preprocessing and model training to
deployment in a user-friendly desktop application. Users can input the sepal and petal measurements of an Iris flower, and
the application predicts whether it belongs to **Setosa**, **Versicolor**, or **Virginica**.

---

## ✨ Features

- 🌸 Predicts Iris flower species using a trained KNN classifier
- 🎨 Modern dark-themed GUI built with CustomTkinter
- 📊 Displays prediction confidence using neighbor voting
- ✅ Validates user inputs before prediction
- ⚡ Performs real-time predictions
- 💾 Loads a pre-trained machine learning model using Pickle
- 📈 Displays trained model accuracy and optimal K value
- 🛡 Handles invalid inputs and missing model files gracefully

---

## 🖥️ Application Preview


## 🛠 Tech Stack

- Python
- CustomTkinter
- Scikit-learn
- NumPy
- Pickle

---

## 🧠 Machine Learning Workflow

- Data Collection
- Data Preprocessing
- Feature Scaling
- Label Encoding
- Train-Test Split
- K-Nearest Neighbors (KNN) Training
- Hyperparameter Selection (Best K)
- Model Serialization using Pickle
- Desktop GUI Deployment

---

## 📂 Project Structure

```
Species-analyzer/
│
├── app.py                 # Desktop application
├── train_model.py         # Model training script
├── model.pkl              # Saved trained model
├── iris.csv               # Dataset
├── requirements.txt
└── README.md
```

---

## 📋 Input Features

The application accepts the following flower measurements:

| Feature | Unit |
|----------|------|
| Sepal Length | cm |
| Sepal Width | cm |
| Petal Length | cm |
| Petal Width | cm |

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/luckypanday11/Species-analyzer.git
```

Navigate to the project directory

```bash
cd Species-analyzer
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## 📊 Model Information

**Algorithm**

- K-Nearest Neighbors (KNN)

**Preprocessing**

- StandardScaler
- Label Encoding

**Prediction Output**

- Predicted Species
- Neighbor Voting Confidence
- Model Accuracy
- Best K Value

---

## 🌼 Supported Species

- Iris Setosa
- Iris Versicolor
- Iris Virginica

---

## 🔍 Input Validation

The application validates:

- Missing inputs
- Invalid numeric values
- Measurements outside the acceptable training range

This helps improve prediction reliability and enhances the user experience.

---

## 💡 Future Improvements

- Add support for multiple ML algorithms
- Display prediction probabilities instead of neighbor voting
- Save prediction history
- Export predictions to CSV
- Deploy as a web application using Streamlit or Flask
- Add flower image classification using deep learning

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a Pull Request

---

## 👨‍💻 Author

**Lucky Panday**
GitHub: https://github.com/luckypanday11

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.
