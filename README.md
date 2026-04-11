# Titanic - Machine Learning from Disaster

A comprehensive machine learning solution for the Kaggle Titanic competition, implementing data preprocessing, feature engineering, and multiple classification models to predict passenger survival.

## Overview

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This project builds a predictive model to determine which passengers were likely to survive based on various features.

## Dataset

The dataset contains passenger information with the following features:

| Feature | Description |
|---------|-------------|
| `PassengerId` | Unique identifier for each passenger |
| `Survived` | Survival (0 = No, 1 = Yes) |
| `Pclass` | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd) |
| `Name` | Passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

### Files

- `dataset/train.csv` - Training data with 891 passengers and survival labels
- `dataset/test.csv` - Test data with 418 passengers for prediction
- `dataset/gender_submission.csv` - Sample submission file

## Project Structure

```
.
├── dataset/                    # Dataset files
│   ├── train.csv              # Training data
│   ├── test.csv               # Test data
│   └── gender_submission.csv  # Sample submission
├── notebooks/                  # Jupyter notebooks
│   └── processing_and_modelling.ipynb  # Main analysis notebook
├── submissions/               # Kaggle submission files
│   ├── submission_LR.csv      # Logistic Regression predictions
│   ├── submission_RF.csv      # Random Forest predictions
│   └── submission_XGB.csv     # XGBoost predictions
├── environment_linux.yml      # Conda environment specification
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.12+
- conda or pip

### Setup Environment

Using conda:
```bash
conda env create -f environment_linux.yml
conda activate titanic
```

Or install dependencies manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

## Methodology

### 1. Data Preprocessing

- **Missing value imputation**: Filled missing Age values with median, Embarked with mode
- **Feature encoding**: Converted categorical variables (Sex, Embarked) to numerical
- **Feature engineering**: Created FamilySize, IsAlone, and Title features from Name
- **Class balancing**: Addressed imbalanced dataset using undersampling techniques

### 2. Exploratory Data Analysis

Key insights discovered:
- **Sex**: Women had significantly higher survival rates (~74% vs ~19%)
- **Pclass**: 1st class passengers had better survival chances (~63% vs ~24% in 3rd class)
- **Age**: Children had higher survival rates
- **FamilySize**: Passengers with small families (2-4 members) had better survival rates

### 3. Feature Engineering

| Engineered Feature | Description |
|-------------------|-------------|
| `FamilySize` | SibSp + Parch + 1 |
| `IsAlone` | Binary flag for solo travelers |
| `Title` | Extracted from passenger names (Mr, Mrs, Miss, Master, etc.) |
| `AgeGroup` | Binned age categories |
| `FarePerPerson` | Fare divided by family size |

### 4. Models Implemented

1. **Logistic Regression** - Baseline linear model
2. **Random Forest** - Ensemble of decision trees
3. **XGBoost** - Gradient boosting with regularization

### 5. Model Evaluation

Models were evaluated using:
- Classification metrics (accuracy, precision, recall, F1-score)
74% accuracy on testing set on submission for the RF classifier

## Usage

### Running the Notebook

```bash
jupyter notebook notebooks/processing_and_modelling.ipynb
```

### Making Predictions

The notebook contains code to generate predictions for the test set:

```python
# Generate submission file
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': predictions
})
submission.to_csv('submissions/submission.csv', index=False)
```

## Key Learnings

1. **Feature engineering is crucial**: Extracting Title from names significantly improved model performance
2. **Handle missing data carefully**: Age imputation using median by Pclass and Sex groups worked better than global median
3. **Family dynamics matter**: IsAlone and FamilySize were among the most important features
4. **Class imbalance**: The dataset is imbalanced (577 vs 314), requiring careful evaluation metrics

## Future Improvements

- [ ] Hyperparameter tuning with GridSearchCV or Optuna
- [ ] Ensemble methods (VotingClassifier, Stacking)
- [ ] Deep learning approaches (neural networks)
- [ ] Feature selection using Recursive Feature Elimination
- [ ] Advanced imputation techniques (KNN, Iterative)

## References

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

- **Pablo** - Machine Learning Enthusiast

---

*This project was created for educational purposes as part of the Kaggle Titanic competition.*
