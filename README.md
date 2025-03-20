# Parkinson's Disease Prediction - Hybrid Model (LSTM + Random Forest)

## Overview
This project aims to predict Parkinson's disease using a hybrid deep learning and machine learning approach. The system consists of an LSTM model for feature extraction and a Random Forest classifier for final predictions. A Flask-based web application is built to accept CSV files and return predictions.

---

## Approach & Methodology
### 1. Data Preprocessing
- The dataset is loaded from `parkinson disease.csv`.
- Exploratory Data Analysis (EDA) is performed, including:
  - Checking for missing values
  - Generating statistical summaries
  - Visualizing the distribution of the target variable (`status`)
  - Computing feature correlations
- Missing values (if any) are filled with the column mean.
- Features (`X`) and target labels (`y`) are separated, with `name` dropped as an irrelevant feature.
- Data is normalized using `MinMaxScaler` to ensure consistent scaling.
- The dataset is split into training and testing sets (80:20 ratio).

### 2. LSTM-Based Feature Extraction
- An LSTM model is designed with:
  - 64 LSTM units (return_sequences=False)
  - A Dense layer (32 neurons, ReLU activation) to produce meaningful feature representations.
- The model is compiled with Adam optimizer and Mean Squared Error (MSE) loss.
- The training process is optimized using `EarlyStopping` to prevent overfitting.
- Once trained, the model is used to extract features from both training and testing data.

### 3. Random Forest Classifier
- The extracted LSTM features are used to train a Random Forest classifier with 100 estimators.
- The trained Random Forest model is used for final predictions.
- The accuracy of the hybrid model is evaluated on the test set.

### 4. Model Saving
- The trained LSTM model is saved as `lstm_model.h5`.
- The trained Random Forest classifier is saved as `random_forest_model.pkl`.
- The MinMaxScaler is saved as `scaler.pkl` to ensure consistent transformations during inference.

---

## Flask Web Application
### Functionality
- The Flask web app allows users to upload a CSV file for prediction.
- The server checks for required features and fills missing columns with zero.
- The uploaded data is preprocessed, scaled, and passed through the LSTM feature extractor.
- Extracted features are then fed to the Random Forest classifier to make predictions.
- Results are returned as JSON output.

### Endpoints
| Route        | Method | Description |
|-------------|--------|-------------|
| `/`         | GET    | Renders the upload page |
| `/predict`  | POST   | Accepts a CSV file, processes it, and returns predictions |

---

## Technologies Used
- **Programming Language:** Python
- **Libraries:**
  - Deep Learning: TensorFlow/Keras
  - Machine Learning: Scikit-learn
  - Data Processing: Pandas, NumPy
  - Visualization: Seaborn, Matplotlib
  - Web Framework: Flask
- **Deployment Considerations:** Flask-based API can be deployed on cloud platforms like AWS, Google Cloud, or Heroku.

---

## Model Performance
- The hybrid model achieves an accuracy of **X%** (replace with actual test accuracy).
- Feature extraction with LSTM helps in improving classification performance.

---

## Future Enhancements
- Fine-tune LSTM architecture for improved feature extraction.
- Implement additional ensemble models to enhance predictive performance.
- Deploy the Flask app as a production-ready web service.

---

## How to Run
1. Install dependencies:  
   ```sh
   pip install -r requirements.txt
   ```
2. Start the Flask app:  
   ```sh
   python app.py
   ```
3. Open `http://127.0.0.1:5000/` in a browser to upload a CSV and make predictions.

---

## Author
- **[Your Name]** (Replace with your actual name)
- Contact: [Your Email] (Optional)

