"""
Ablation Study Framework for Systematic Experimentation
"""
import numpy as np
import pandas as pd
from itertools import product
from typing import Dict, List, Any
import json
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from pipeline import ExamScorePipeline, prepare_features
from sklearn.model_selection import train_test_split


class AblationStudy:
    """
    Framework for running systematic ablation studies on the regression pipeline.
    """
    
    def __init__(self, data_path: str, random_state: int = 42):
        """
        Initialize ablation study.
        
        Args:
            data_path: Path to the processed dataset
            random_state: Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.results = []
        
        # Set random seeds
        np.random.seed(random_state)
        
    def define_experiments(self) -> List[Dict[str, Any]]:
        """
        Define all ablation experiments to run.
        
        Returns:
            List of experiment configurations
        """
        # Baseline configuration
        baseline = {
            'name': 'baseline',
            'scaler_type': 'standard',
            'encoder_type': 'onehot',
            'model_type': 'randomforest',
            'model_kwargs': {'n_estimators': 200, 'max_depth': None}
        }
        
        experiments = [baseline]
        
        # Ablation 1: Scaler variations
        for scaler in ['none', 'minmax', 'robust']:
            exp = baseline.copy()
            exp['name'] = f'scaler_{scaler}'
            exp['scaler_type'] = scaler
            experiments.append(exp)
        
        # Ablation 2: Encoder variations
        exp = baseline.copy()
        exp['name'] = 'encoder_ordinal'
        exp['encoder_type'] = 'ordinal'
        experiments.append(exp)
        
        # Ablation 3: Model variations
        model_configs = {
            'gradientboosting': {'n_estimators': 200, 'learning_rate': 0.1},
            'linear': {},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 0.1},
            'knn': {'n_neighbors': 5}
        }
        
        for model_name, kwargs in model_configs.items():
            exp = baseline.copy()
            exp['name'] = f'model_{model_name}'
            exp['model_type'] = model_name
            exp['model_kwargs'] = kwargs
            experiments.append(exp)
        
        # Ablation 4: RandomForest hyperparameter variations
        rf_variants = [
            {'n_estimators': 100, 'max_depth': None},
            {'n_estimators': 300, 'max_depth': None},
            {'n_estimators': 200, 'max_depth': 10},
            {'n_estimators': 200, 'max_depth': 20}
        ]
        
        for i, kwargs in enumerate(rf_variants):
            exp = baseline.copy()
            exp['name'] = f'rf_variant_{i+1}'
            exp['model_kwargs'] = kwargs
            experiments.append(exp)
        
        # Ablation 5: Combined variations
        combinations = [
            {'scaler_type': 'minmax', 'encoder_type': 'ordinal', 'model_type': 'gradientboosting'},
            {'scaler_type': 'robust', 'encoder_type': 'onehot', 'model_type': 'gradientboosting'},
            {'scaler_type': 'none', 'encoder_type': 'onehot', 'model_type': 'knn'}
        ]
        
        for i, config in enumerate(combinations):
            exp = baseline.copy()
            exp['name'] = f'combined_{i+1}'
            exp.update(config)
            if config['model_type'] == 'gradientboosting':
                exp['model_kwargs'] = {'n_estimators': 200, 'learning_rate': 0.1}
            elif config['model_type'] == 'knn':
                exp['model_kwargs'] = {'n_neighbors': 5}
            experiments.append(exp)
        
        return experiments
    
    def run_single_experiment(
        self,
        config: Dict[str, Any],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        numeric_features: List[str],
        categorical_features: List[str]
    ) -> Dict[str, Any]:
        """
        Run a single experiment with given configuration.
        
        Args:
            config: Experiment configuration
            X_train, X_test, y_train, y_test: Train/test splits
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            
        Returns:
            Dictionary with experiment results
        """
        print(f"\nRunning experiment: {config['name']}")
        print(f"  Scaler: {config['scaler_type']}")
        print(f"  Encoder: {config['encoder_type']}")
        print(f"  Model: {config['model_type']}")
        
        # Initialize pipeline
        pipeline = ExamScorePipeline(
            scaler_type=config['scaler_type'],
            encoder_type=config['encoder_type'],
            model_type=config['model_type'],
            random_state=self.random_state,
            **config.get('model_kwargs', {})
        )
        
        # Build and fit pipeline
        pipeline.build_pipeline(numeric_features, categorical_features)
        
        # Time training
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate on test set
        test_metrics = pipeline.evaluate(X_test, y_test)
        
        # Time inference
        start_time = time.time()
        _ = pipeline.predict(X_test)
        inference_time = time.time() - start_time
        
        # Get feature importance if available
        feature_importance = None
        if config['model_type'] in ['randomforest', 'gradientboosting']:
            importance_df = pipeline.get_feature_importance()
            if importance_df is not None:
                feature_importance = importance_df.head(10).to_dict('records')
        
        # Compile results
        result = {
            'experiment_name': config['name'],
            'config': config,
            'metrics': test_metrics,
            'training_time': training_time,
            'inference_time': inference_time,
            'feature_importance': feature_importance
        }
        
        print(f"  Results: MSE={test_metrics['mse']:.2f}, R²={test_metrics['r2']:.4f}, Time={training_time:.2f}s")
        
        return result
    
    def run_all_experiments(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        save_models: bool = True,
        artifacts_dir: str = '../artifacts'
    ):
        """
        Run all defined ablation experiments.
        
        Args:
            df: Full dataset
            test_size: Proportion for test split
            save_models: Whether to save trained models
            artifacts_dir: Directory to save models
        """
        # Prepare features
        numeric_features, categorical_features = prepare_features(df)
        
        # Prepare data - drop columns as per main.ipynb notebook
        # These columns were excluded in the final model training
        columns_to_drop = [
            'exam_score', 'student_id', 'gender', 'diet_score', 
            'extracurricular_participation', 'age', 'part_job_score', 
            'part_time_job', 'internet_quality', 'internet_score', 
            'parental_education_level', 'diet_quality', 'mental_health_rating'
        ]
        X = df.drop(columns=columns_to_drop, errors='ignore')
        y = df['exam_score']
        
        # Split data (fixed split for fair comparison)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"Dataset split: {len(X_train)} train, {len(X_test)} test")
        print(f"Numeric features ({len(numeric_features)}): {numeric_features}")
        print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        
        # Define and run experiments
        experiments = self.define_experiments()
        print(f"\nTotal experiments to run: {len(experiments)}")
        
        self.results = []
        
        for config in experiments:
            try:
                result = self.run_single_experiment(
                    config, X_train, X_test, y_train, y_test,
                    numeric_features, categorical_features
                )
                self.results.append(result)
                
                # Save model if requested
                if save_models:
                    artifacts_path = Path(artifacts_dir)
                    artifacts_path.mkdir(parents=True, exist_ok=True)
                    
                    # Rebuild pipeline to save
                    pipeline = ExamScorePipeline(
                        scaler_type=config['scaler_type'],
                        encoder_type=config['encoder_type'],
                        model_type=config['model_type'],
                        random_state=self.random_state,
                        **config.get('model_kwargs', {})
                    )
                    pipeline.build_pipeline(numeric_features, categorical_features)
                    pipeline.fit(X_train, y_train)
                    
                    model_path = artifacts_path / f"{config['name']}_model.joblib"
                    pipeline.save(str(model_path))
                    
            except Exception as e:
                print(f"  ERROR in {config['name']}: {str(e)}")
                continue
        
        print("\n" + "="*50)
        print("All experiments completed!")
        print(f"Successful: {len(self.results)}/{len(experiments)}")
        
        return self.results
    
    def save_results(self, output_dir: str = '../experiments'):
        """
        Save experiment results to JSON and CSV.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save full results as JSON
        json_path = output_path / 'ablation_results.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create summary DataFrame
        summary_data = []
        for result in self.results:
            summary_data.append({
                'experiment': result['experiment_name'],
                'scaler': result['config']['scaler_type'],
                'encoder': result['config']['encoder_type'],
                'model': result['config']['model_type'],
                'mse': result['metrics']['mse'],
                'r2': result['metrics']['r2'],
                'mae': result['metrics']['mae'],
                'rmse': result['metrics']['rmse'],
                'training_time': result['training_time'],
                'inference_time': result['inference_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('r2', ascending=False)
        
        # Save as CSV
        csv_path = output_path / 'ablation_summary.csv'
        summary_df.to_csv(csv_path, index=False)
        
        print(f"\nResults saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        
        return summary_df
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary of all experiments as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        for result in self.results:
            summary_data.append({
                'experiment': result['experiment_name'],
                'scaler': result['config']['scaler_type'],
                'encoder': result['config']['encoder_type'],
                'model': result['config']['model_type'],
                'mse': result['metrics']['mse'],
                'r2': result['metrics']['r2'],
                'mae': result['metrics']['mae'],
                'rmse': result['metrics']['rmse'],
                'training_time': result['training_time']
            })
        
        return pd.DataFrame(summary_data).sort_values('r2', ascending=False)
