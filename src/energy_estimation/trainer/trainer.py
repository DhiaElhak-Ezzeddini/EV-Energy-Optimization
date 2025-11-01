"""
Model Trainer with Optuna Optimization

This module provides a comprehensive training framework for EV energy estimation models,
including hyperparameter optimization using Optuna, logging, and model management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)
import json
import logging
from datetime import datetime
import joblib
from time import perf_counter
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.energy_estimation.models.base_model import BaseModel
from src.energy_estimation.data_utils.preprocessor import Preprocessor


class ModelTrainer:
    """
    Comprehensive trainer for energy estimation models with Optuna optimization.
    
    Features:
    - Hyperparameter optimization using Optuna
    - Comprehensive logging
    - Model checkpointing
    - Metric tracking
    - Visualization support
    - Cross-validation support
    
    Attributes:
        model (BaseModel): The model to train
        save_dir (Path): Directory to save models and logs
        experiment_name (str): Name of the experiment
        logger (logging.Logger): Logger instance
    """
    
    def __init__(
        self,
        model: BaseModel,
        save_dir: Optional[str] = None,
        experiment_name: Optional[str] = None,
        verbose: bool = True,
        preprocessor: Optional[Preprocessor] = None,
    ):
        """
        Initialize the model trainer.
        
        Args:
            model (BaseModel): Model instance to train
            save_dir (Optional[str]): Directory to save models and logs.
                If None, uses ../saved_models relative to this file.
            experiment_name (Optional[str]): Name for this experiment
            verbose (bool): Whether to print detailed logs
            preprocessor (Optional[Preprocessor]): Preprocessor for inverse-transforming targets
        """
        self.model = model
        
        # Set save directory - use absolute path to avoid confusion
        if save_dir is None:
            # Default: save to src/energy_estimation/saved_models
            save_dir = Path(__file__).parent.parent / "saved_models"
        
        self.save_dir = Path(save_dir).resolve()  # resolve to absolute path
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment name
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{model.model_name}_{timestamp}"
        self.experiment_name = experiment_name
        
        # Create experiment directory
        self.experiment_dir = self.save_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.verbose = verbose
        self.logger = self._setup_logger()
        
        # Training history
        self.training_history: Dict[str, Any] = {
            'experiment_name': experiment_name,
            'model_name': model.model_name,
            'start_time': None,
            'end_time': None,
            'duration': None,
            'best_params': {},
            'best_score': None,
            'metrics': {},
            'optuna_study': None
        }
        
        self.logger.info(f"Trainer initialized for experiment: {experiment_name}")
        # Optional preprocessor used to inverse-transform target for metrics
        self.preprocessor = preprocessor
    
    def _setup_logger(self) -> logging.Logger:
        """
        Setup logger for the trainer.
        
        Returns:
            logging.Logger: Configured logger
        """
        # Create logger
        logger = logging.getLogger(f"trainer_{self.experiment_name}")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler
        log_file = self.experiment_dir / "training.log"
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_optuna: bool = True,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        direction: str = "minimize",
        metric: str = "rmse",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model with or without Optuna optimization.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            use_optuna (bool): Whether to use Optuna for hyperparameter optimization
            n_trials (int): Number of Optuna trials
            timeout (Optional[int]): Timeout in seconds for Optuna
            direction (str): Optimization direction ('minimize' or 'maximize')
            metric (str): Metric to optimize ('rmse', 'mae', 'r2', 'mape')
            **kwargs: Additional training parameters
            
        Returns:
            Dict[str, Any]: Training results including metrics and history
        """
        self.logger.info("="*70)
        self.logger.info(f"Starting training for {self.model.model_name}")
        self.logger.info("="*70)
        
        start_time = datetime.now()
        self.training_history['start_time'] = start_time.isoformat()
        
        # Log data shapes
        self.logger.info(f"Training data: {X_train.shape}")
        self.logger.info(f"Validation data: {X_val.shape}")
        self.logger.info(f"Use Optuna: {use_optuna}")
        
        try:
            # Train with or without Optuna
            if use_optuna:
                results = self._train_with_optuna(
                    X_train, y_train, X_val, y_val,
                    n_trials=n_trials,
                    timeout=timeout,
                    direction=direction,
                    metric=metric,
                    **kwargs
                )
            else:
                results = self._train_without_optuna(
                    X_train, y_train, X_val, y_val,
                    metric=metric,
                    **kwargs
                )
            
            # Calculate final metrics
            self.logger.info("\nCalculating final metrics...")
            final_metrics = self.evaluate(X_val, y_val)
            results['final_metrics'] = final_metrics
            
            # Update training history
            end_time = datetime.now()
            self.training_history['end_time'] = end_time.isoformat()
            self.training_history['duration'] = str(end_time - start_time)
            self.training_history['metrics'] = final_metrics
            self.training_history['best_params'] = self.model.best_params
            
            # Save model and results
            self.logger.info("\nSaving model and results...")
            self._save_training_results(results)
            
            self.logger.info("="*70)
            self.logger.info("Training completed successfully!")
            self.logger.info("="*70)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}", exc_info=True)
            raise
    
    def _train_with_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        direction: str = "minimize",
        metric: str = "rmse",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train model with Optuna hyperparameter optimization.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of Optuna trials
            timeout: Timeout in seconds
            direction: Optimization direction
            metric: Metric to optimize
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Training results
        """
        self.logger.info(f"\nStarting Optuna optimization with {n_trials} trials...")
        self.logger.info(f"Optimizing metric: {metric} ({direction})")
        
        # Train the model (this will create the Optuna study internally)
        training_results = self.model.train(
            X_train, y_train,
            X_val, y_val,
            use_optuna=True,
            n_trials=n_trials,
            timeout=timeout,
            **kwargs
        )
        
        # Store study for later analysis
        if hasattr(self.model, 'study') and self.model.study is not None:
            self.training_history['optuna_study'] = self.model.study
            
            # Log best trial info
            best_trial = self.model.study.best_trial
            self.logger.info(f"\nBest trial: {best_trial.number}")
            self.logger.info(f"Best value: {best_trial.value:.6f}")
            self.logger.info("Best parameters:")
            for key, value in best_trial.params.items():
                self.logger.info(f"  {key}: {value}")
            
            # Save Optuna visualizations
            self._save_optuna_visualizations(self.model.study)
        
        return training_results
    
    def _train_without_optuna(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str = "rmse",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train model without Optuna (using default or provided parameters).
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            metric: Metric to track
            **kwargs: Additional parameters including model params
            
        Returns:
            Dict[str, Any]: Training results
        """
        self.logger.info("\nTraining without Optuna optimization...")
        
        # Extract model parameters if provided
        model_params = kwargs.get('params', None)
        
        # Build and train model
        self.model.build(params=model_params)
        training_results = self.model.train(
            X_train, y_train,
            X_val, y_val,
            use_optuna=False,
            **kwargs
        )
        
        return training_results
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        set_name: str = "validation"
    ) -> Dict[str, float]:
        """
        Evaluate model on given data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): True targets
            set_name (str): Name of the dataset (for logging)
            
        Returns:
            Dict[str, float]: Evaluation metrics including inference time
        """
        self.logger.info(f"\nEvaluating on {set_name} set...")
        
        # Time the prediction
        start_time = perf_counter()
        y_pred = self.model.predict(X)
        inference_time = perf_counter() - start_time
        
        # Calculate inference speed metrics
        samples_per_second = len(X) / inference_time if inference_time > 0 else 0
        time_per_sample_ms = (inference_time / len(X)) * 1000 if len(X) > 0 else 0

        # If a preprocessor with target scaling was provided, inverse-transform
        # both predictions and true values back to original scale before metrics
        if self.preprocessor is not None and getattr(self.preprocessor, 'scale_target', False):
            try:
                y_pred = self.preprocessor.inverse_transform_target(np.asarray(y_pred))
            except Exception:
                # Fallback: leave as-is if inverse fails
                self.logger.warning("Could not inverse-transform y_pred using preprocessor")

            try:
                y = self.preprocessor.inverse_transform_target(np.asarray(y))
            except Exception:
                self.logger.warning("Could not inverse-transform y (true target) using preprocessor")

        # Calculate metrics
        metrics = self._calculate_metrics(y, y_pred)
        
        # Add timing metrics
        metrics['inference_time_seconds'] = inference_time
        metrics['samples_per_second'] = samples_per_second
        metrics['time_per_sample_ms'] = time_per_sample_ms
        
        # Log metrics
        self.logger.info(f"\n{set_name.capitalize()} Metrics:")
        for metric_name, value in metrics.items():
            if 'time' in metric_name or 'samples' in metric_name:
                if 'seconds' in metric_name:
                    self.logger.info(f"  {metric_name}: {value:.6f} s")
                elif 'ms' in metric_name:
                    self.logger.info(f"  {metric_name}: {value:.6f} ms")
                else:
                    self.logger.info(f"  {metric_name}: {value:.2f}")
            else:
                self.logger.info(f"  {metric_name}: {value:.6f}")
        
        return metrics
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.
        
        Args:
            y_true (np.ndarray): True values
            y_pred (np.ndarray): Predicted values
            
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,  # Convert to percentage
        }
        
        # Calculate additional metrics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        metrics['max_error'] = np.max(np.abs(residuals))
        
        return metrics
    
    def _save_optuna_visualizations(self, study: optuna.Study) -> None:
        """
        Save Optuna study visualizations.
        
        Args:
            study (optuna.Study): Optuna study object
        """
        try:
            viz_dir = self.experiment_dir / "optuna_visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            self.logger.info("\nSaving Optuna visualizations...")
            
            # Optimization history
            try:
                fig = plot_optimization_history(study)
                fig.write_html(str(viz_dir / "optimization_history.html"))
                self.logger.debug("  - Optimization history saved")
            except Exception as e:
                self.logger.warning(f"  - Could not save optimization history: {e}")
            
            # Parameter importances
            try:
                fig = plot_param_importances(study)
                fig.write_html(str(viz_dir / "param_importances.html"))
                self.logger.debug("  - Parameter importances saved")
            except Exception as e:
                self.logger.warning(f"  - Could not save parameter importances: {e}")
            
            # Parallel coordinate plot
            try:
                fig = plot_parallel_coordinate(study)
                fig.write_html(str(viz_dir / "parallel_coordinate.html"))
                self.logger.debug("  - Parallel coordinate plot saved")
            except Exception as e:
                self.logger.warning(f"  - Could not save parallel coordinate plot: {e}")
            
            self.logger.info(f"Visualizations saved to: {viz_dir}")
            
        except Exception as e:
            self.logger.warning(f"Could not save visualizations: {e}")
    
    def _save_training_results(self, results: Dict[str, Any]) -> None:
        """
        Save training results, model, and metadata.
        
        Args:
            results (Dict[str, Any]): Training results
        """
        # Save model
        model_path = self.model.save_model(
            save_dir=str(self.experiment_dir),
            include_study=True,
            metadata={
                'experiment_name': self.experiment_name,
                'training_results': results
            }
        )
        
        # Save training history
        history_path = self.experiment_dir / "training_history.json"
        with open(history_path, 'w') as f:
            # Convert non-serializable objects
            history_copy = self.training_history.copy()
            if 'optuna_study' in history_copy:
                del history_copy['optuna_study']  # Don't save study in JSON
            json.dump(history_copy, f, indent=4, default=str)
        
        self.logger.info(f"Training history saved to: {history_path}")
        
        # Save summary
        summary_path = self.experiment_dir / "summary.txt"
        self._save_summary(summary_path, results)
    
    def _save_summary(self, path: Path, results: Dict[str, Any]) -> None:
        """
        Save a human-readable summary of the training.
        
        Args:
            path (Path): Path to save the summary
            results (Dict[str, Any]): Training results
        """
        with open(path, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"Training Summary: {self.experiment_name}\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Model: {self.model.model_name}\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Start Time: {self.training_history['start_time']}\n")
            f.write(f"End Time: {self.training_history['end_time']}\n")
            f.write(f"Duration: {self.training_history['duration']}\n\n")
            
            f.write("Best Parameters:\n")
            f.write("-"*70 + "\n")
            for key, value in self.model.best_params.items():
                f.write(f"  {key}: {value}\n")
            
            if 'final_metrics' in results:
                f.write("\nFinal Metrics:\n")
                f.write("-"*70 + "\n")
                for key, value in results['final_metrics'].items():
                    f.write(f"  {key}: {value:.6f}\n")
            
            f.write("\n" + "="*70 + "\n")
        
        self.logger.info(f"Summary saved to: {path}")
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        use_optuna: bool = False,
        n_trials: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv (int): Number of folds
            use_optuna (bool): Whether to optimize hyperparameters for each fold
            n_trials (int): Number of Optuna trials per fold
            **kwargs: Additional parameters
            
        Returns:
            Dict[str, Any]: Cross-validation results
        """
        from sklearn.model_selection import KFold
        
        self.logger.info(f"\nPerforming {cv}-fold cross-validation...")
        
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_results = {
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X), 1):
            self.logger.info(f"\nFold {fold}/{cv}")
            self.logger.info("-"*50)
            
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Train on fold
            if use_optuna:
                self.model.train(
                    X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold,
                    use_optuna=True,
                    n_trials=n_trials,
                    **kwargs
                )
            else:
                self.model.train(
                    X_train_fold, y_train_fold,
                    X_val_fold, y_val_fold,
                    use_optuna=False,
                    **kwargs
                )
            
            # Evaluate fold
            fold_metrics = self.evaluate(X_val_fold, y_val_fold, set_name=f"Fold {fold}")
            cv_results['fold_metrics'].append(fold_metrics)
        
        # Calculate mean and std across folds
        metrics_df = pd.DataFrame(cv_results['fold_metrics'])
        cv_results['mean_metrics'] = metrics_df.mean().to_dict()
        cv_results['std_metrics'] = metrics_df.std().to_dict()
        
        # Log CV summary
        self.logger.info("\n" + "="*70)
        self.logger.info("Cross-Validation Summary")
        self.logger.info("="*70)
        for metric in cv_results['mean_metrics'].keys():
            mean_val = cv_results['mean_metrics'][metric]
            std_val = cv_results['std_metrics'][metric]
            self.logger.info(f"{metric}: {mean_val:.6f} (+/- {std_val:.6f})")
        
        return cv_results
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            top_n (Optional[int]): Return only top N features
            
        Returns:
            pd.DataFrame: Feature importance DataFrame
        """
        importance = self.model.get_feature_importance()
        
        if importance is not None and top_n is not None:
            importance = importance.head(top_n)
        
        return importance
    
    def predict_and_compare(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        set_name: str = "test"
    ) -> pd.DataFrame:
        """
        Generate predictions and return a DataFrame comparing predictions with ground truth.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Ground truth targets
            set_name (str): Name of the dataset (for logging)
            
        Returns:
            pd.DataFrame: DataFrame with columns ['actual', 'predicted', 'residual', 'abs_error', 'pct_error']
        """
        self.logger.info(f"\nGenerating predictions for {set_name} set...")
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # If a preprocessor with target scaling was provided, inverse-transform
        # both predictions and true values back to original scale
        if self.preprocessor is not None and getattr(self.preprocessor, 'scale_target', False):
            try:
                y_pred = self.preprocessor.inverse_transform_target(np.asarray(y_pred))
            except Exception:
                self.logger.warning("Could not inverse-transform y_pred using preprocessor")
            
            try:
                y_actual = self.preprocessor.inverse_transform_target(np.asarray(y))
            except Exception:
                self.logger.warning("Could not inverse-transform y (true target) using preprocessor")
                y_actual = np.asarray(y)
        else:
            y_actual = np.asarray(y)
        
        # Create comparison DataFrame
        results_df = pd.DataFrame({
            'actual': y_actual,
            'predicted': y_pred,
            'residual': y_actual - y_pred,
            'abs_error': np.abs(y_actual - y_pred),
            'pct_error': np.abs((y_actual - y_pred) / y_actual) * 100
        })
        
        # Reset index to match original data if possible
        if isinstance(y, pd.Series):
            results_df.index = y.index
        
        self.logger.info(f"Prediction comparison DataFrame created with {len(results_df)} rows")
        
        return results_df
    
    def save_predictions(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        set_name: str = "test",
        filename: Optional[str] = None
    ) -> str:
        """
        Generate predictions and save comparison DataFrame to CSV.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Ground truth targets
            set_name (str): Name of the dataset
            filename (Optional[str]): Custom filename, if None uses default
            
        Returns:
            str: Path where the predictions were saved
        """
        # Generate predictions
        results_df = self.predict_and_compare(X, y, set_name)
        
        # Determine save path
        if filename is None:
            filename = f"{set_name}_predictions.csv"
        
        save_path = self.experiment_dir / filename
        
        # Save to CSV
        results_df.to_csv(save_path, index=True)
        
        self.logger.info(f"Predictions saved to: {save_path}")
        
        return str(save_path)
    
    def __repr__(self) -> str:
        """String representation of the trainer."""
        return f"ModelTrainer(model={self.model.model_name}, experiment={self.experiment_name})"
