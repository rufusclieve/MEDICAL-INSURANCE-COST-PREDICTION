# Medical Insurance Cost Prediction

This repository contains the implementation of a Machine Learning project for predicting medical insurance costs based on several factors such as age, BMI, gender, and region. The goal of this project is to build a predictive model that can estimate the insurance cost for a new customer based on their personal and medical details.

## Project Overview

Medical insurance cost prediction is an essential aspect for insurance companies as it helps them estimate premiums and understand customer risk profiles. This project leverages machine learning techniques to create a model that can accurately predict insurance costs. The dataset used contains features like:

- **Age**: The age of the individual
- **Sex**: Gender of the individual (male or female)
- **BMI**: Body Mass Index
- **Children**: Number of children/dependents
- **Smoker**: Whether the individual is a smoker (yes or no)
- **Region**: Residential area of the individual
- **Charges**: The medical insurance cost (target variable)

## Dataset

The dataset contains the following key columns:
- **age**: Age of the individual
- **sex**: Gender
- **bmi**: Body Mass Index
- **children**: Number of children the individual has
- **smoker**: Whether the individual is a smoker or not
- **region**: The region where the individual lives
- **charges**: The actual medical insurance cost charged

## Machine Learning Models

In this project, multiple regression models were implemented to predict insurance costs:

1. **Linear Regression**: A basic regression model used as a baseline.
2. **Random Forest Regression**: An ensemble learning method that improves prediction accuracy.
3. **Lasso Regression**: A regularized regression technique that adds a penalty to prevent overfitting.

## Project Structure

The project is organized as follows:

- `data/`: Contains the dataset used for training the model.
- `notebooks/`: Jupyter notebooks showcasing the data preprocessing, EDA, and model training.
- `src/`: Source code for model implementation, training, and evaluation.
- `README.md`: Project description and details (this file).

## Requirements

To run this project, you'll need the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install the required dependencies using:
```
pip install -r requirements.txt
```

## Running the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-insurance-prediction.git
   ```

2. Navigate to the project directory:
   ```bash
   cd medical-insurance-prediction
   ```

3. Run the Jupyter notebook or Python script to train and test the models:
   ```bash
   jupyter notebook notebooks/Medical_Insurance_Cost_Prediction.ipynb
   ```

## Model Performance

- **Linear Regression**: Basic model with reasonable accuracy.

Evaluation metrics used:
- R-squared (R²)

## Conclusion

This project demonstrates the process of building and evaluating regression models to predict medical insurance costs. The Random Forest model provided the most accurate results, and the Lasso Regression helped in feature selection and reducing overfitting.
