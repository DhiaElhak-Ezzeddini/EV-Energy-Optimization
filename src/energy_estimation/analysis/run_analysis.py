"""
Run Performance Analysis on EV Energy Consumption Models.

This script runs comprehensive analysis on a random sample of the dataset,
comparing LightGBM, CatBoost, XGBoost, and Stacking ensemble models.

Usage:
    python run_analysis.py [--samples N] [--show-plots]
    
Examples:
    python run_analysis.py                    # 500 samples, no interactive plots
    python run_analysis.py --samples 1000     # 1000 samples
    python run_analysis.py --show-plots       # Show plots interactively
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from analysis.performance_analyzer import PerformanceAnalyzer


def main():
    """Main entry point for analysis."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive performance analysis on EV energy models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      Run with default 500 samples
  %(prog)s --samples 1000       Run with 1000 random samples
  %(prog)s --show-plots         Display plots interactively
  %(prog)s --samples 200 --show-plots
        """
    )
    
    parser.add_argument(
        '--samples', '-n',
        type=int,
        default=500,
        help='Number of random samples to use (default: 500)'
    )
    
    parser.add_argument(
        '--show-plots', '-p',
        action='store_true',
        help='Display plots interactively (default: False, just save)'
    )
    
    parser.add_argument(
        '--test-size', '-t',
        type=float,
        default=0.2,
        help='Test set fraction (default: 0.2)'
    )
    
    parser.add_argument(
        '--random-state', '-r',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for results (default: analysis/results)'
    )
    
    args = parser.parse_args()
    
    # Determine data path (go up to project root)
    project_root = Path(__file__).parent.parent.parent.parent
    data_path = project_root / "data" / "EV_Energy_Consumption_Dataset.csv"
    
    if not data_path.exists():
        print(f"❌ Error: Dataset not found at {data_path}")
        print(f"   Please ensure the dataset is available.")
        sys.exit(1)
    
    # Create analyzer
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    analyzer = PerformanceAnalyzer(
        data_path=data_path,
        n_samples=args.samples,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=output_dir
    )
    
    # Run analysis
    try:
        results = analyzer.run_full_analysis(show_plots=args.show_plots)
        
        print("\n" + "=" * 70)
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"\n📊 Summary:")
        print(f"   - Models analyzed: {len(results['models'])}")
        print(f"   - Samples used: {args.samples}")
        print(f"   - Test samples: {len(analyzer.y_test)}")
        print(f"   - Visualizations: {len(results['plot_paths'])}")
        print(f"\n📁 Results location: {results['output_dir']}")
        print(f"\n📄 Files generated:")
        print(f"   - analysis_report.csv")
        for name, path in results['plot_paths'].items():
            print(f"   - {path.name}")
        
        # Print top model
        best_model = results['report'].index[0]
        best_metrics = results['results'][best_model]
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   RMSE: {best_metrics['rmse']:.4f}")
        print(f"   MAE: {best_metrics['mae']:.4f}")
        print(f"   R²: {best_metrics['r2']:.4f}")
        print(f"   MAPE: {best_metrics['mape']:.2f}%")
        print(f"   Speed: {best_metrics['samples_per_second']:,.0f} samples/sec")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
