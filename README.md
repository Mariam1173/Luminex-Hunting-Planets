# LUMINEX: Hunting for Exoplanets

## Overview

LUMINEX is an AI-driven project aimed at enhancing the detection and classification of exoplanets using data from NASA's TESS mission. By leveraging machine learning models, we aim to distinguish true exoplanet signals from astrophysical and instrumental false positives, thereby improving the efficiency of astronomical research.

## Project Highlights

- **AI Accuracy**: Achieved over 85% accuracy in classifying exoplanet candidates.
- **Dual Approach**: Combined classical machine learning models with deep learning techniques.
- **Open Source**: Code and models are available for public use and further development.

## Team Members

- **Mariam Tawfik** – AeroSpace and AI Engineer
- **Tasneem Abdallah** – AeroSpace Engineer
- **Mohamed Yasser** - AeroSpace Engineer
- **Menna Emam** – AeroSpace Engineer
- **Mostafa Sameh** – AeroSpace Engineer

## Technologies Used

- **Python**: Primary programming language.
- **pandas**: Data manipulation and analysis.
- **scikit-learn**: Machine learning algorithms.
- **TensorFlow/Keras**: Deep learning models.
- **joblib**: Model serialization.

## Methodology

1. **Data Preparation**:
   - Loaded and preprocessed TESS data using pandas.
   - Filtered and balanced dataset to focus on Planet Candidates (PC) and False Positives (FP).
   - Scaled numerical features and encoded labels.

2. **Model Development**:
   - Implemented and evaluated classical models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVC, KNN, XGBoost.
   - Developed a Multi-Layer Perceptron (MLP) using TensorFlow/Keras.

3. **Evaluation Metrics**:
   - Assessed models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
   - Compared performance visually through bar charts and plots.

4. **Model Selection**:
   - Identified the best-performing model based on evaluation metrics.
   - Saved the selected model using joblib for future use.

## Results

- **MLP Model**:
  - AUC: 0.813
  - Average Precision: 0.932
  - High recall (~96%), minimizing false negatives.

- **Classical Models**:
  - Gradient Boosting and Random Forest showed balanced performance across key metrics.

## Conclusion

LUMINEX provides a robust framework for exoplanet detection, combining traditional and modern machine learning approaches. Our models offer a reliable tool for astronomers to filter out false positives and focus on genuine exoplanet candidates, accelerating the pace of discovery.

## Getting Started

To run the models locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Mariam1173/Luminex-Hunting-Planets.git
   cd LUMINEX

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the StreamLit App:
   ```bash
   streamlit run app.py


