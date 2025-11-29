"""
Configurable Regression Pipeline for Exam Score Prediction
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from typing import Dict, Any, Tuple, Optional


class ExamScorePipeline:
    """
    Flexible pipeline for exam score prediction with configurable preprocessing and models.
    """
    
    def __init__(
        self,
        scaler_type: str = 'standard',
        encoder_type: str = 'onehot',
        model_type: str = 'randomforest',
        random_state: int = 42,
        **model_kwargs
    ):
        """
        Initialize the pipeline with specified components.
        
        Args:
            scaler_type: 'standard', 'minmax', 'robust', or 'none'
            encoder_type: 'onehot' or 'ordinal'
            model_type: 'randomforest', 'gradientboosting', 'linear', 'ridge', 'lasso', 'svr', 'knn'
            random_state: Random seed for reproducibility
            **model_kwargs: Additional parameters for the model
        """
        self.scaler_type = scaler_type
        self.encoder_type = encoder_type
        self.model_type = model_type
        self.random_state = random_state
        self.model_kwargs = model_kwargs
        self.pipeline = None
        self.feature_names = None
        self.numeric_features = None
        self.categorical_features = None
        
    def _get_scaler(self):
        """Return scaler based on type."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'none': 'passthrough'
        }
        return scalers.get(self.scaler_type, StandardScaler())
    
    def _get_encoder(self):
        """Return encoder based on type."""
        encoders = {
            'onehot': OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            'ordinal': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        }
        return encoders.get(self.encoder_type, OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    
    def _get_model(self):
        """Return model based on type."""
        models = {
            'randomforest': RandomForestRegressor(random_state=self.random_state, **self.model_kwargs),
            'gradientboosting': GradientBoostingRegressor(random_state=self.random_state, **self.model_kwargs),
            'linear': LinearRegression(**self.model_kwargs),
            'ridge': Ridge(random_state=self.random_state, **self.model_kwargs),
            'lasso': Lasso(random_state=self.random_state, **self.model_kwargs),
            'svr': SVR(**self.model_kwargs),
            'knn': KNeighborsRegressor(**self.model_kwargs)
        }
        return models.get(self.model_type, RandomForestRegressor(random_state=self.random_state))
    
    def build_pipeline(self, numeric_features: list, categorical_features: list):
        """
        Build the complete pipeline with preprocessing and model.
        
        Args:
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
        """
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        # Numeric preprocessing
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', self._get_scaler())
        ])
        
        # Categorical preprocessing
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', self._get_encoder())
        ])
        
        # Combine preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Build full pipeline
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', self._get_model())
        ])
        
        return self.pipeline
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit the pipeline on training data."""
        if self.pipeline is None:
            raise ValueError("Pipeline not built. Call build_pipeline() first.")
        
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame):
        """Make predictions on new data."""
        if self.pipeline is None:
            raise ValueError("Pipeline not built or fitted.")
        
        return self.pipeline.predict(X)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the pipeline on test data.
        
        Returns:
            Dictionary with MSE, R2, and MAE metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred))
        }
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Returns:
            Dictionary with mean and std of scores
        """
        scores = cross_val_score(
            self.pipeline, X, y, 
            cv=cv, 
            scoring='r2',
            n_jobs=-1
        )
        
        return {
            'cv_r2_mean': scores.mean(),
            'cv_r2_std': scores.std(),
            'cv_scores': scores
        }
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance for tree-based models.
        
        Returns:
            DataFrame with feature names and importance scores, or None if not applicable
        """
        model = self.pipeline.named_steps['regressor']
        
        if hasattr(model, 'feature_importances_'):
            # Get feature names after preprocessing
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            # Get feature names from transformers
            feature_names = []
            
            # Numeric features
            feature_names.extend(self.numeric_features)
            
            # Categorical features (after encoding)
            if self.encoder_type == 'onehot':
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
                if hasattr(cat_encoder, 'get_feature_names_out'):
                    cat_features = cat_encoder.get_feature_names_out(self.categorical_features)
                    feature_names.extend(cat_features)
                else:
                    feature_names.extend(self.categorical_features)
            else:
                feature_names.extend(self.categorical_features)
            
            importance_df = pd.DataFrame({
                'feature': feature_names[:len(model.feature_importances_)],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        
        return None
    
    def save(self, filepath: str):
        """Save the pipeline to disk."""
        joblib.dump(self.pipeline, filepath)
        
    @classmethod
    def load(cls, filepath: str):
        """Load a saved pipeline."""
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        return instance


def prepare_features(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Identify numeric and categorical features from dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    # Exclude target, ID columns, and features dropped in main.ipynb
    exclude_cols = [
        'exam_score', 'student_id', 'gender', 'diet_score', 
        'extracurricular_participation', 'age', 'part_job_score', 
        'part_time_job', 'internet_quality', 'internet_score', 
        'parental_education_level', 'diet_quality', 'mental_health_rating'
    ]
    
    # Identify feature types
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove excluded columns
    numeric_features = [col for col in numeric_features if col not in exclude_cols]
    categorical_features = [col for col in categorical_features if col not in exclude_cols]
    
    return numeric_features, categorical_features
