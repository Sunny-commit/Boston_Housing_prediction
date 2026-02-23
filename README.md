# 🚗 Boston Housing Prediction - ML Regression

An **end-to-end machine learning project predicting house prices** using the Boston Housing dataset, demonstrating regression techniques, feature engineering, and model evaluation for real estate market analysis.

## 🎯 Overview

This project covers:
- ✅ Data preprocessing & cleaning
- ✅ Exploratory data analysis
- ✅ Feature engineering & selection
- ✅ Multiple regression models
- ✅ Hyperparameter tuning
- ✅ Model evaluation

## 📊 Dataset Overview

```
Boston Housing Dataset:
├── 506 properties
├── 13 input features
├── 1 target (MEDV - Median home value)
├── Features:
│   ├── CRIM - Crime rate
│   ├── ZN - Zoned residential land
│   ├── INDUS - Non-retail business
│   ├── CHAS - Charles River dummy variable
│   ├── NOX - Nitrogen oxide concentration
│   ├── RM - Rooms per dwelling
│   ├── AGE - Building age
│   ├── DIS - Distance to employment centers
│   ├── RAD - Accessibility to highways
│   ├── TAX - Property tax rate
│   ├── PTRATIO - Pupil-teacher ratio
│   ├── BLACK - Proportion of Black population
│   └── LSTAT - Lower status population %
```

## 🔍 Exploratory Data Analysis

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

class BostonHousingEDA:
    """Analyze Boston Housing data"""
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.target = 'MEDV'
    
    def basic_statistics(self):
        """Get dataset statistics"""
        stats = self.data.describe()
        
        print("Dataset Shape:", self.data.shape)
        print("\nBasic Statistics:\n", stats)
        print("\nMissing Values:\n", self.data.isnull().sum())
        
        return stats
    
    def correlation_analysis(self):
        """Analyze feature correlations"""
        corr_matrix = self.data.corr()
        
        # Correlation with target
        target_corr = corr_matrix[self.target].sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.show()
        
        return target_corr
    
    def feature_distributions(self):
        """Analyze feature distributions"""
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        
        for idx, col in enumerate(self.data.columns[:-1]):
            ax = axes[idx // 4, idx % 4]
            ax.hist(self.data[col], bins=30, edgecolor='black')
            ax.set_title(col)
        
        plt.tight_layout()
        plt.show()
    
    def outlier_detection(self):
        """Identify outliers using IQR"""
        outliers = {}
        
        for col in self.data.columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            outliers[col] = outlier_mask.sum()
        
        return outliers
```

## 🔧 Feature Engineering

```python
class FeatureEngineer:
    """Create and select features"""
    
    def __init__(self, df):
        self.df = df.copy()
    
    def create_polynomial_features(self, features, degree=2):
        """Create polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(self.df[features])
        
        feature_names = poly.get_feature_names_out(features)
        
        return pd.DataFrame(poly_features, columns=feature_names)
    
    def create_interaction_features(self):
        """Create interaction terms"""
        self.df['RM_squared'] = self.df['RM'] ** 2  # Room count effect
        self.df['LSTAT_RM_interaction'] = self.df['LSTAT'] * self.df['RM']
        self.df['NOX_DIS_interaction'] = self.df['NOX'] * self.df['DIS']
        
        return self.df
    
    def create_binned_features(self):
        """Create categorical from continuous"""
        self.df['AGE_GROUP'] = pd.cut(self.df['AGE'], bins=4)
        self.df['RM_CATEGORY'] = pd.cut(self.df['RM'], bins=3)
        
        return self.df
    
    def feature_scaling(self):
        """Normalize features"""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df)
        
        return pd.DataFrame(scaled_data, columns=self.df.columns), scaler
```

## 🤖 Multiple Regression Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

class RegressionModels:
    """Train multiple regression models"""
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.models = {}
    
    def linear_regression(self):
        """Basic linear regression"""
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        
        self.models['Linear'] = model
        return model
    
    def ridge_regression(self, alpha=1.0):
        """Ridge regression (L2 regularization)"""
        model = Ridge(alpha=alpha)
        model.fit(self.X_train, self.y_train)
        
        self.models['Ridge'] = model
        return model
    
    def lasso_regression(self, alpha=0.1):
        """Lasso regression (L1 regularization)"""
        model = Lasso(alpha=alpha, max_iter=5000)
        model.fit(self.X_train, self.y_train)
        
        self.models['Lasso'] = model
        return model
    
    def random_forest(self, n_trees=100):
        """Random forest regression"""
        model = RandomForestRegressor(n_estimators=n_trees, random_state=42)
        model.fit(self.X_train, self.y_train)
        
        self.models['RF'] = model
        return model
    
    def gradient_boosting(self):
        """Gradient boosting regressor"""
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        model.fit(self.X_train, self.y_train)
        
        self.models['GB'] = model
        return model
    
    def svr(self, kernel='rbf', C=100):
        """Support Vector Regression"""
        model = SVR(kernel=kernel, C=C)
        model.fit(self.X_train, self.y_train)
        
        self.models['SVR'] = model
        return model
    
    def compare_models(self, X_test, y_test):
        """Compare model performance"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            results[name] = {
                'MAE': np.mean(np.abs(y_test - y_pred)),
                'RMSE': np.sqrt(np.mean((y_test - y_pred) ** 2)),
                'R2': model.score(X_test, y_test)
            }
        
        return pd.DataFrame(results).T
```

## 🔍 Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

class HyperparameterTuning:
    """Optimize model hyperparameters"""
    
    @staticmethod
    def grid_search_rf(X_train, y_train):
        """Grid search for Random Forest"""
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_model': grid_search.best_estimator_
        }
    
    @staticmethod
    def learning_curves(model, X_train, y_train, cv=5):
        """Plot learning curves"""
        from sklearn.model_selection import learning_curve
        
        train_sizes, train_scores, val_scores = learning_curve(
            model,
            X_train,
            y_train,
            cv=cv,
            n_jobs=-1
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validation')
        plt.xlabel('Training Set Size')
        plt.ylabel('R² Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.show()
```

## 📊 Model Evaluation

```python
class ModelEvaluation:
    """Evaluate model performance"""
    
    @staticmethod
    def residual_analysis(y_true, y_pred):
        """Analyze prediction residuals"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Residual plot
        axes[0].scatter(y_pred, residuals)
        axes[0].axhline(y=0, color='r', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residual Plot')
        
        # Distribution
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        
        plt.tight_layout()
        plt.show()
        
        return residuals
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate evaluation metrics"""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        return {
            'MAE': mean_absolute_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
```

## 💡 Interview Talking Points

**Q: Why use regularization (Ridge/Lasso)?**
```
Answer:
- Prevent overfitting
- Ridge (L2): Shrink coefficients slightly
- Lasso (L1): Feature selection (zero coefficients)
```

**Q: How select best model?**
```
Answer:
- Cross-validation (CV score)
- Test set evaluation
- Business constraints (interpretability vs accuracy)
```

## 🌟 Portfolio Value

✅ End-to-end ML project
✅ Regression techniques
✅ Feature engineering
✅ Model comparison & selection
✅ Hyperparameter tuning
✅ Professional ML pipeline

---

**Technologies**: Scikit-learn, Pandas, NumPy, Matplotlib

