"""
Performance Analyzer for EV Energy Consumption Models.

Provides comprehensive analysis including training, evaluation, and visualization
on a random sample of the dataset.
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import models
try:
    # Relative imports (when used as package)
    from ..models.lightgbm_model import LightGBMModel
    from ..models.catboost_model import CatBoostModel
    from ..models.xgboost_model import XGBoostModel
    from ..models.stacking_model import StackingEnsembleModel
    from ..data_loader.dataset_loader import DatasetLoader
except ImportError:
    # Absolute imports (when run as script)
    import sys
    parent_dir = Path(__file__).parent.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    from models.lightgbm_model import LightGBMModel
    from models.catboost_model import CatBoostModel
    from models.xgboost_model import XGBoostModel
    from models.stacking_model import StackingEnsembleModel
    from data_loader.dataset_loader import DatasetLoader


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for EV energy consumption models.
    
    Loads random sample, trains models, collects metrics, and generates visualizations.
    """
    
    def __init__(
        self,
        data_path: Path,
        n_samples: int = 500,
        test_size: float = 0.2,
        random_state: int = 42,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the analyzer.
        
        Args:
            data_path: Path to the dataset CSV
            n_samples: Number of random samples to use
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            output_dir: Directory to save results and plots
        """
        self.data_path = Path(data_path)
        self.n_samples = n_samples
        self.test_size = test_size
        self.random_state = random_state
        
        if output_dir is None:
            self.output_dir = Path(__file__).parent / "results"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize attributes
        self.data_loader: Optional[DatasetLoader] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.feature_names: Optional[List[str]] = None
        
        self.models: Dict[str, Any] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        
        print(f"📊 Performance Analyzer initialized")
        print(f"   Dataset: {self.data_path}")
        print(f"   Sample size: {self.n_samples}")
        print(f"   Test size: {self.test_size:.1%}")
        print(f"   Output directory: {self.output_dir}")
    
    def load_sample_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and prepare random sample from dataset.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print(f"\n📂 Loading random sample of {self.n_samples} from dataset...")
        
        # Load full dataset
        self.data_loader = DatasetLoader(
            data_path=str(self.data_path),
            use_preprocessor=False  # We'll handle preprocessing separately
        )
        df = self.data_loader.load_data(drop_identifiers=True)
        
        print(f"   Original dataset: {df.shape[0]} samples, {df.shape[1]} features")
        
        # Random sample
        if self.n_samples < df.shape[0]:
            df = df.sample(n=self.n_samples, random_state=self.random_state)
            print(f"   Sampled: {self.n_samples} random samples")
        else:
            print(f"   Using all {df.shape[0]} samples (requested sample size >= dataset size)")
        
        # Separate features and target
        self.feature_names = self.data_loader.feature_columns
        X = df[self.feature_names]
        y = df[self.data_loader.target_feature]
        
        # Split data
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Store as DataFrames/Series
        self.X_train = X_train_temp.reset_index(drop=True)
        self.X_test = X_test_temp.reset_index(drop=True)
        self.y_train = y_train_temp.reset_index(drop=True)
        self.y_test = y_test_temp.reset_index(drop=True)
        
        print(f"   Train set: {self.X_train.shape[0]} samples")
        print(f"   Test set: {self.X_test.shape[0]} samples")
        print(f"   Features: {', '.join(self.feature_names[:5])}..." if len(self.feature_names) > 5 else f"   Features: {', '.join(self.feature_names)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Train all models on the sample data.
        
        Returns:
            Dictionary of trained models
        """
        if self.X_train is None:
            raise ValueError("Data not loaded. Call load_sample_data() first.")
        
        print(f"\n🚀 Training all models...")
        print("=" * 60)
        
        # Define models
        model_configs = {
            'LightGBM': LightGBMModel(),
            'CatBoost': CatBoostModel(),
            'XGBoost': XGBoostModel(),
            'Stacking': StackingEnsembleModel()
        }
        
        for name, model in model_configs.items():
            print(f"\n📦 Training {name}...")
            start_time = time.time()
            
            model.train(self.X_train, self.y_train)
            
            train_time = time.time() - start_time
            self.models[name] = model
            
            print(f"   ✓ Training completed in {train_time:.2f} seconds")
        
        print("\n" + "=" * 60)
        print(f"✅ All {len(self.models)} models trained successfully!")
        
        return self.models
    
    def collect_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Collect comprehensive metrics for all models.
        
        Returns:
            Dictionary mapping model names to their metrics
        """
        if not self.models:
            raise ValueError("Models not trained. Call train_all_models() first.")
        
        print(f"\n📈 Collecting metrics...")
        print("=" * 60)
        
        # Convert test data to numpy for metrics calculation
        y_test_array = self.y_test.values
        
        for name, model in self.models.items():
            print(f"\n🔍 Evaluating {name}...")
            
            # Make predictions and time it
            start_time = time.time()
            y_pred = model.predict(self.X_test)
            inference_time = time.time() - start_time
            
            # Convert predictions to numpy array if not already
            if hasattr(y_pred, 'values'):
                y_pred = y_pred.values
            
            # Store predictions
            self.predictions[name] = y_pred
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test_array, y_pred))
            mae = mean_absolute_error(y_test_array, y_pred)
            mse = mean_squared_error(y_test_array, y_pred)
            r2 = r2_score(y_test_array, y_pred)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test_array - y_pred) / y_test_array)) * 100
            
            # Inference performance
            samples_per_second = len(self.X_test) / inference_time
            
            self.results[name] = {
                'rmse': rmse,
                'mae': mae,
                'mse': mse,
                'r2': r2,
                'mape': mape,
                'inference_time_seconds': inference_time,
                'samples_per_second': samples_per_second,
                'time_per_sample_ms': (inference_time / len(self.X_test)) * 1000
            }
            
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   R²: {r2:.4f}")
            print(f"   MAPE: {mape:.2f}%")
            print(f"   Inference: {inference_time*1000:.2f} ms ({samples_per_second:,.0f} samples/sec)")
        
        print("\n" + "=" * 60)
        print("✅ Metrics collected for all models!")
        
        return self.results
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """
        Extract feature importance from all models.
        
        Returns:
            Dictionary mapping model names to feature importance dicts
        """
        importance_dict = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'get_feature_importance'):
                importance = model.get_feature_importance()
                
                # Map indices to feature names
                if self.feature_names:
                    importance_dict[name] = {
                        self.feature_names[i]: score 
                        for i, score in enumerate(importance)
                    }
                else:
                    importance_dict[name] = {
                        f'Feature_{i}': score 
                        for i, score in enumerate(importance)
                    }
        
        return importance_dict
    
    def generate_visualizations(self, show: bool = False) -> Dict[str, Path]:
        """
        Generate all visualization plots.
        
        Args:
            show: Whether to display plots interactively
            
        Returns:
            Dictionary mapping plot names to saved file paths
        """
        if not self.results:
            raise ValueError("No results available. Call collect_metrics() first.")
        
        print(f"\n📊 Generating visualizations...")
        print("=" * 60)
        
        from .visualization import (
            plot_error_metrics_comparison,
            plot_inference_time_comparison,
            plot_residuals_analysis,
            plot_prediction_scatter,
            plot_feature_importance,
            plot_error_distribution
        )
        
        plot_paths = {}
        
        # Convert y_test to numpy for visualization
        y_test_array = self.y_test.values
        
        # 1. Error Metrics Comparison
        print("\n📉 Creating error metrics comparison...")
        path = self.output_dir / "error_metrics_comparison.png"
        plot_error_metrics_comparison(self.results, save_path=path, show=show)
        plot_paths['error_metrics'] = path
        
        # 2. Inference Time Comparison
        print("⏱️  Creating inference time comparison...")
        path = self.output_dir / "inference_time_comparison.png"
        plot_inference_time_comparison(self.results, save_path=path, show=show)
        plot_paths['inference_time'] = path
        
        # 3. Residuals Analysis
        print("📊 Creating residuals analysis...")
        path = self.output_dir / "residuals_analysis.png"
        plot_residuals_analysis(y_test_array, self.predictions, save_path=path, show=show)
        plot_paths['residuals'] = path
        
        # 4. Prediction Scatter
        print("🎯 Creating prediction scatter plots...")
        path = self.output_dir / "prediction_scatter.png"
        plot_prediction_scatter(y_test_array, self.predictions, save_path=path, show=show)
        plot_paths['scatter'] = path
        
        # 5. Feature Importance
        print("🔍 Creating feature importance plots...")
        importance_dict = self.get_feature_importance()
        if importance_dict:
            path = self.output_dir / "feature_importance.png"
            plot_feature_importance(importance_dict, save_path=path, show=show)
            plot_paths['feature_importance'] = path
        
        # 6. Error Distribution
        print("📈 Creating error distribution plots...")
        path = self.output_dir / "error_distribution.png"
        plot_error_distribution(y_test_array, self.predictions, save_path=path, show=show)
        plot_paths['error_distribution'] = path
        
        print("\n" + "=" * 60)
        print(f"✅ All visualizations saved to: {self.output_dir}")
        
        return plot_paths
    
    def generate_report(self, save_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Generate comprehensive analysis report.
        
        Args:
            save_path: Path to save the report CSV
            
        Returns:
            DataFrame with all metrics
        """
        if not self.results:
            raise ValueError("No results available. Call collect_metrics() first.")
        
        print(f"\n📋 Generating analysis report...")
        
        # Create DataFrame
        df = pd.DataFrame(self.results).T
        
        # Round for readability
        df = df.round({
            'rmse': 4,
            'mae': 4,
            'mse': 4,
            'r2': 4,
            'mape': 2,
            'inference_time_seconds': 4,
            'samples_per_second': 0,
            'time_per_sample_ms': 3
        })
        
        # Sort by RMSE (best first)
        df = df.sort_values('rmse')
        
        # Add ranking
        df.insert(0, 'rank', range(1, len(df) + 1))
        
        # Save if requested
        if save_path is None:
            save_path = self.output_dir / "analysis_report.csv"
        
        df.to_csv(save_path)
        print(f"   ✓ Report saved to: {save_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        print(df.to_string())
        print("=" * 60)
        
        # Highlight best model
        best_model = df.index[0]
        best_rmse = df.loc[best_model, 'rmse']
        best_r2 = df.loc[best_model, 'r2']
        
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   RMSE: {best_rmse:.4f}")
        print(f"   R²: {best_r2:.4f}")
        print(f"   MAPE: {df.loc[best_model, 'mape']:.2f}%")
        
        return df
    
    def run_full_analysis(self, show_plots: bool = False) -> Dict[str, Any]:
        """
        Run complete analysis pipeline.
        
        Args:
            show_plots: Whether to display plots interactively
            
        Returns:
            Dictionary with all analysis results
        """
        print("\n" + "=" * 70)
        print("🚀 STARTING FULL PERFORMANCE ANALYSIS")
        print("=" * 70)
        
        # Step 1: Load data
        self.load_sample_data()
        
        # Step 2: Train models
        self.train_all_models()
        
        # Step 3: Collect metrics
        self.collect_metrics()
        
        # Step 4: Generate visualizations
        plot_paths = self.generate_visualizations(show=show_plots)
        
        # Step 5: Generate report
        report_df = self.generate_report()
        
        print("\n" + "=" * 70)
        print("✅ FULL ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📁 All results saved to: {self.output_dir}")
        print(f"   - Analysis report: analysis_report.csv")
        print(f"   - Plots: {len(plot_paths)} visualization files")
        
        return {
            'models': self.models,
            'predictions': self.predictions,
            'results': self.results,
            'report': report_df,
            'plot_paths': plot_paths,
            'output_dir': self.output_dir
        }


if __name__ == "__main__":
    # Example usage
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / "data" / "EV_Energy_Consumption_Dataset.csv"
    
    analyzer = PerformanceAnalyzer(
        data_path=data_path,
        n_samples=500,
        test_size=0.2,
        random_state=42
    )
    
    results = analyzer.run_full_analysis(show_plots=False)
    
    print(f"\n✅ Analysis complete! Results saved to: {results['output_dir']}")
