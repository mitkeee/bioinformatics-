#!/usr/bin/env python3
"""
DUDE Dataset Analysis - Complete Implementation
Processes 100+ proteins from DUDE dataset with:
- 2 confusion matrices per protein (vs DSSP and STRIDE)
- Cross-validation (5-fold or 10-fold)
- Parameter optimization using Optuna
- Per-protein and whole-dataset accuracy
- Outlier detection and analysis
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
import json
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from comprehensive_burial_analysis import (
    BurialParameters,
    extract_ca_atoms,
    extract_dssp_data,
    extract_stride_data,
    add_neighbor_features,
    classify_burial,
    ProteinResults
)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


@dataclass
class DUDEAnalysisConfig:
    """Configuration for DUDE dataset analysis"""
    # Cross-validation
    n_folds: int = 5  # 5-fold or 10-fold CV
    train_ratio: float = 0.8  # 80% training, 20% validation
    
    # Optimization
    n_optimization_trials: int = 100
    optimization_reference: str = 'dssp'  # 'dssp' or 'stride'
    
    # Output
    output_base_dir: Path = Path("results/dude_analysis")
    save_individual_csvs: bool = True
    save_confusion_matrices: bool = True


class DUDEDatasetAnalyzer:
    """
    Complete analyzer for DUDE dataset with cross-validation and optimization
    """
    
    def __init__(self, pdb_directory: Path, config: DUDEAnalysisConfig = None):
        self.pdb_dir = pdb_directory
        self.config = config or DUDEAnalysisConfig()
        self.output_dir = self.config.output_base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all PDB files
        self.pdb_files = self._find_pdb_files()
        print(f"Found {len(self.pdb_files)} PDB files in {pdb_directory}")
        
    def _find_pdb_files(self) -> List[Path]:
        """Find all PDB files in directory and subdirectories"""
        pdb_files = []
        
        # Check main directory
        pdb_files.extend(self.pdb_dir.glob("*.pdb"))
        pdb_files.extend(self.pdb_dir.glob("*.ent"))
        
        # Check subdirectories
        for subdir in self.pdb_dir.iterdir():
            if subdir.is_dir():
                pdb_files.extend(subdir.glob("*.pdb"))
                pdb_files.extend(subdir.glob("*.ent"))
        
        return sorted(list(set(pdb_files)))
    
    def process_single_protein(self, pdb_path: Path, params: BurialParameters) -> Optional[ProteinResults]:
        """Process a single protein and return results"""
        try:
            protein_id = pdb_path.stem
            
            # Extract CA atoms
            df = extract_ca_atoms(pdb_path)
            if len(df) == 0:
                print(f"  Warning: No CA atoms found in {protein_id}")
                return None
            
            coords = df[['x', 'y', 'z']].values
            
            # Extract reference data
            df = extract_dssp_data(pdb_path, df, params.dssp_asa_cutoff)
            df = extract_stride_data(pdb_path, df, params.stride_asa_cutoff)
            
            # Add neighbor features
            df = add_neighbor_features(df, coords)
            
            # Classify using our algorithm
            df['ncps_class'] = classify_burial(df, params)
            
            # Calculate metrics vs DSSP
            dssp_mask = df['dssp_class'].notna()
            dssp_metrics = None
            
            if dssp_mask.sum() > 0:
                y_true = df.loc[dssp_mask, 'dssp_class'].values.astype(int)
                y_pred = df.loc[dssp_mask, 'ncps_class'].values.astype(int)
                
                dssp_metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0)
                }
            
            # Calculate metrics vs STRIDE
            stride_mask = df['stride_class'].notna()
            stride_metrics = None
            
            if stride_mask.sum() > 0:
                y_true = df.loc[stride_mask, 'stride_class'].values.astype(int)
                y_pred = df.loc[stride_mask, 'ncps_class'].values.astype(int)
                
                stride_metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'f1': f1_score(y_true, y_pred, zero_division=0)
                }
            
            return ProteinResults(
                protein_id=protein_id,
                n_residues=len(df),
                dataframe=df,
                dssp_accuracy=dssp_metrics['accuracy'] if dssp_metrics else None,
                dssp_confusion_matrix=dssp_metrics['confusion_matrix'] if dssp_metrics else None,
                dssp_precision=dssp_metrics['precision'] if dssp_metrics else None,
                dssp_recall=dssp_metrics['recall'] if dssp_metrics else None,
                dssp_f1=dssp_metrics['f1'] if dssp_metrics else None,
                stride_accuracy=stride_metrics['accuracy'] if stride_metrics else None,
                stride_confusion_matrix=stride_metrics['confusion_matrix'] if stride_metrics else None,
                stride_precision=stride_metrics['precision'] if stride_metrics else None,
                stride_recall=stride_metrics['recall'] if stride_metrics else None,
                stride_f1=stride_metrics['f1'] if stride_metrics else None
            )
            
        except Exception as e:
            print(f"  Error processing {pdb_path.stem}: {e}")
            return None
    
    def run_analysis_with_params(self, params: BurialParameters, protein_subset: List[Path] = None) -> Dict:
        """
        Run analysis on proteins with given parameters
        Returns dict with results and aggregate statistics
        """
        proteins_to_process = protein_subset if protein_subset else self.pdb_files
        
        results = []
        for i, pdb_file in enumerate(proteins_to_process, 1):
            print(f"Processing {i}/{len(proteins_to_process)}: {pdb_file.stem}...", end='\r')
            result = self.process_single_protein(pdb_file, params)
            if result:
                results.append(result)
        
        print()  # New line after progress
        
        # Calculate aggregate statistics
        stats = self._calculate_aggregate_stats(results)
        
        return {
            'results': results,
            'statistics': stats,
            'parameters': params
        }
    
    def _calculate_aggregate_stats(self, results: List[ProteinResults]) -> Dict:
        """Calculate aggregate statistics across all proteins"""
        stats = {
            'n_proteins': len(results),
            'total_residues': sum(r.n_residues for r in results),
        }
        
        # DSSP statistics
        dssp_accuracies = [r.dssp_accuracy for r in results if r.dssp_accuracy is not None]
        dssp_f1_scores = [r.dssp_f1 for r in results if r.dssp_f1 is not None]
        
        if dssp_accuracies:
            # Aggregate confusion matrix
            dssp_cm_total = np.zeros((2, 2), dtype=int)
            for r in results:
                if r.dssp_confusion_matrix is not None:
                    dssp_cm_total += r.dssp_confusion_matrix
            
            stats['dssp'] = {
                'mean_accuracy': float(np.mean(dssp_accuracies)),
                'std_accuracy': float(np.std(dssp_accuracies)),
                'min_accuracy': float(np.min(dssp_accuracies)),
                'max_accuracy': float(np.max(dssp_accuracies)),
                'median_accuracy': float(np.median(dssp_accuracies)),
                'mean_f1': float(np.mean(dssp_f1_scores)),
                'std_f1': float(np.std(dssp_f1_scores)),
                'aggregate_confusion_matrix': dssp_cm_total.tolist(),
                'n_proteins_with_data': len(dssp_accuracies)
            }
        
        # STRIDE statistics
        stride_accuracies = [r.stride_accuracy for r in results if r.stride_accuracy is not None]
        stride_f1_scores = [r.stride_f1 for r in results if r.stride_f1 is not None]
        
        if stride_accuracies:
            # Aggregate confusion matrix
            stride_cm_total = np.zeros((2, 2), dtype=int)
            for r in results:
                if r.stride_confusion_matrix is not None:
                    stride_cm_total += r.stride_confusion_matrix
            
            stats['stride'] = {
                'mean_accuracy': float(np.mean(stride_accuracies)),
                'std_accuracy': float(np.std(stride_accuracies)),
                'min_accuracy': float(np.min(stride_accuracies)),
                'max_accuracy': float(np.max(stride_accuracies)),
                'median_accuracy': float(np.median(stride_accuracies)),
                'mean_f1': float(np.mean(stride_f1_scores)),
                'std_f1': float(np.std(stride_f1_scores)),
                'aggregate_confusion_matrix': stride_cm_total.tolist(),
                'n_proteins_with_data': len(stride_accuracies)
            }
        
        return stats
    
    def cross_validate(self, params: BurialParameters, n_folds: int = 5) -> Dict:
        """
        Perform k-fold cross-validation
        Splits proteins into k folds (not individual residues)
        """
        print(f"\n{'='*80}")
        print(f"CROSS-VALIDATION ({n_folds}-FOLD)")
        print(f"{'='*80}\n")
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(self.pdb_files), 1):
            print(f"\nFold {fold_idx}/{n_folds}")
            print("-" * 40)
            
            test_files = [self.pdb_files[i] for i in test_idx]
            print(f"Testing on {len(test_files)} proteins")
            
            # Run analysis on test set
            fold_result = self.run_analysis_with_params(params, test_files)
            fold_results.append(fold_result)
            
            # Print fold statistics
            if 'dssp' in fold_result['statistics']:
                dssp_acc = fold_result['statistics']['dssp']['mean_accuracy']
                dssp_f1 = fold_result['statistics']['dssp']['mean_f1']
                print(f"  DSSP - Accuracy: {dssp_acc:.4f}, F1: {dssp_f1:.4f}")
            
            if 'stride' in fold_result['statistics']:
                stride_acc = fold_result['statistics']['stride']['mean_accuracy']
                stride_f1 = fold_result['statistics']['stride']['mean_f1']
                print(f"  STRIDE - Accuracy: {stride_acc:.4f}, F1: {stride_f1:.4f}")
        
        # Calculate cross-validation statistics
        cv_stats = self._calculate_cv_stats(fold_results)
        
        print(f"\n{'='*80}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*80}\n")
        
        if 'dssp' in cv_stats:
            print(f"DSSP Results:")
            print(f"  Mean Accuracy: {cv_stats['dssp']['mean_accuracy']:.4f} ± {cv_stats['dssp']['std_accuracy']:.4f}")
            print(f"  Mean F1-Score: {cv_stats['dssp']['mean_f1']:.4f} ± {cv_stats['dssp']['std_f1']:.4f}")
        
        if 'stride' in cv_stats:
            print(f"\nSTRIDE Results:")
            print(f"  Mean Accuracy: {cv_stats['stride']['mean_accuracy']:.4f} ± {cv_stats['stride']['std_accuracy']:.4f}")
            print(f"  Mean F1-Score: {cv_stats['stride']['mean_f1']:.4f} ± {cv_stats['stride']['std_f1']:.4f}")
        
        return {
            'fold_results': fold_results,
            'cv_statistics': cv_stats,
            'parameters': params
        }
    
    def _calculate_cv_stats(self, fold_results: List[Dict]) -> Dict:
        """Calculate statistics across cross-validation folds"""
        cv_stats = {}
        
        # DSSP statistics
        dssp_fold_accuracies = []
        dssp_fold_f1s = []
        for fold in fold_results:
            if 'dssp' in fold['statistics']:
                dssp_fold_accuracies.append(fold['statistics']['dssp']['mean_accuracy'])
                dssp_fold_f1s.append(fold['statistics']['dssp']['mean_f1'])
        
        if dssp_fold_accuracies:
            cv_stats['dssp'] = {
                'mean_accuracy': float(np.mean(dssp_fold_accuracies)),
                'std_accuracy': float(np.std(dssp_fold_accuracies)),
                'fold_accuracies': dssp_fold_accuracies,
                'mean_f1': float(np.mean(dssp_fold_f1s)),
                'std_f1': float(np.std(dssp_fold_f1s)),
                'fold_f1s': dssp_fold_f1s
            }
        
        # STRIDE statistics
        stride_fold_accuracies = []
        stride_fold_f1s = []
        for fold in fold_results:
            if 'stride' in fold['statistics']:
                stride_fold_accuracies.append(fold['statistics']['stride']['mean_accuracy'])
                stride_fold_f1s.append(fold['statistics']['stride']['mean_f1'])
        
        if stride_fold_accuracies:
            cv_stats['stride'] = {
                'mean_accuracy': float(np.mean(stride_fold_accuracies)),
                'std_accuracy': float(np.std(stride_fold_accuracies)),
                'fold_accuracies': stride_fold_accuracies,
                'mean_f1': float(np.mean(stride_fold_f1s)),
                'std_f1': float(np.std(stride_fold_f1s)),
                'fold_f1s': stride_fold_f1s
            }
        
        return cv_stats
    
    def optimize_parameters(self, n_trials: int = 100, reference: str = 'dssp') -> BurialParameters:
        """
        Optimize parameters using Optuna with cross-validation
        """
        if not HAS_OPTUNA:
            print("ERROR: Optuna not installed. Using default parameters.")
            return BurialParameters()
        
        print(f"\n{'='*80}")
        print(f"PARAMETER OPTIMIZATION - {n_trials} TRIALS")
        print(f"{'='*80}\n")
        print(f"Optimizing against: {reference.upper()}")
        print(f"Cross-validation: {self.config.n_folds}-fold\n")
        
        def objective(trial):
            """Optuna objective function"""
            # Suggest parameters
            params = BurialParameters(
                nc6_threshold=trial.suggest_float('nc6_threshold', 6.0, 15.0),
                nc10_threshold=trial.suggest_float('nc10_threshold', 12.0, 30.0),
                uni6_threshold=trial.suggest_float('uni6_threshold', 0.25, 0.65),
                uni10_threshold=trial.suggest_float('uni10_threshold', 0.35, 0.75)
            )
            
            # Evaluate using cross-validation
            cv_result = self.cross_validate(params, self.config.n_folds)
            
            # Return mean accuracy
            if reference in cv_result['cv_statistics']:
                return cv_result['cv_statistics'][reference]['mean_accuracy']
            else:
                return 0.0
        
        # Create study and optimize
        study = optuna.create_study(direction='maximize', study_name='dude_optimization')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n{'='*80}")
        print("OPTIMIZATION RESULTS")
        print(f"{'='*80}\n")
        print(f"Best Accuracy: {study.best_value:.4f}")
        print(f"\nBest Parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value:.4f}")
        
        # Save optimization results
        study_df = study.trials_dataframe()
        study_df.to_csv(self.output_dir / "optimization_trials.csv", index=False)
        print(f"\nOptimization trials saved to: {self.output_dir / 'optimization_trials.csv'}")
        
        # Return best parameters
        best_params = BurialParameters(
            nc6_threshold=study.best_params['nc6_threshold'],
            nc10_threshold=study.best_params['nc10_threshold'],
            uni6_threshold=study.best_params['uni6_threshold'],
            uni10_threshold=study.best_params['uni10_threshold']
        )
        
        return best_params
    
    def save_results(self, analysis_result: Dict, output_subdir: str = "final"):
        """Save comprehensive results"""
        output_path = self.output_dir / output_subdir
        output_path.mkdir(exist_ok=True)
        
        results = analysis_result['results']
        stats = analysis_result['statistics']
        params = analysis_result['parameters']
        
        print(f"\nSaving results to: {output_path}")
        
        # Save individual confusion matrices
        if self.config.save_confusion_matrices:
            cm_dir = output_path / "confusion_matrices"
            cm_dir.mkdir(exist_ok=True)
            
            for result in results:
                # DSSP confusion matrix
                if result.dssp_confusion_matrix is not None:
                    cm_df = pd.DataFrame(
                        result.dssp_confusion_matrix,
                        index=['True_Interior(0)', 'True_Exterior(1)'],
                        columns=['Pred_Interior(0)', 'Pred_Exterior(1)']
                    )
                    cm_df.to_csv(cm_dir / f"{result.protein_id}_confusion_matrix_dssp.csv")
                
                # STRIDE confusion matrix
                if result.stride_confusion_matrix is not None:
                    cm_df = pd.DataFrame(
                        result.stride_confusion_matrix,
                        index=['True_Interior(0)', 'True_Exterior(1)'],
                        columns=['Pred_Interior(0)', 'Pred_Exterior(1)']
                    )
                    cm_df.to_csv(cm_dir / f"{result.protein_id}_confusion_matrix_stride.csv")
        
        # Save individual protein CSVs
        if self.config.save_individual_csvs:
            csv_dir = output_path / "protein_csvs"
            csv_dir.mkdir(exist_ok=True)
            
            for result in results:
                result.dataframe.to_csv(csv_dir / f"{result.protein_id}_detailed.csv", index=False)
        
        # Save aggregate statistics
        with open(output_path / "aggregate_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save parameters
        params_dict = {
            'nc6_threshold': params.nc6_threshold,
            'nc10_threshold': params.nc10_threshold,
            'uni6_threshold': params.uni6_threshold,
            'uni10_threshold': params.uni10_threshold,
            'dssp_asa_cutoff': params.dssp_asa_cutoff,
            'stride_asa_cutoff': params.stride_asa_cutoff
        }
        with open(output_path / "parameters.json", 'w') as f:
            json.dump(params_dict, f, indent=2)
        
        # Save summary report
        self._save_summary_report(results, stats, params, output_path / "summary_report.txt")
        
        # Save per-protein accuracy table
        self._save_per_protein_table(results, output_path / "per_protein_accuracy.csv")
        
        print(f"✓ Results saved successfully")
    
    def _save_summary_report(self, results: List[ProteinResults], stats: Dict, 
                            params: BurialParameters, filepath: Path):
        """Generate and save comprehensive summary report"""
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DUDE DATASET ANALYSIS - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Parameters
            f.write("PARAMETERS USED:\n")
            f.write("-" * 40 + "\n")
            f.write(f"nc6_threshold: {params.nc6_threshold:.4f}\n")
            f.write(f"nc10_threshold: {params.nc10_threshold:.4f}\n")
            f.write(f"uni6_threshold: {params.uni6_threshold:.4f}\n")
            f.write(f"uni10_threshold: {params.uni10_threshold:.4f}\n")
            f.write(f"dssp_asa_cutoff: {params.dssp_asa_cutoff:.4f}\n")
            f.write(f"stride_asa_cutoff: {params.stride_asa_cutoff:.4f}\n\n")
            
            # Dataset overview
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Proteins Analyzed: {stats['n_proteins']}\n")
            f.write(f"Total Residues: {stats['total_residues']}\n\n")
            
            # DSSP results
            if 'dssp' in stats:
                f.write("="*80 + "\n")
                f.write("RESULTS vs DSSP\n")
                f.write("="*80 + "\n\n")
                f.write(f"Proteins with DSSP data: {stats['dssp']['n_proteins_with_data']}\n")
                f.write(f"Mean Accuracy: {stats['dssp']['mean_accuracy']:.4f} ± {stats['dssp']['std_accuracy']:.4f}\n")
                f.write(f"Median Accuracy: {stats['dssp']['median_accuracy']:.4f}\n")
                f.write(f"Min Accuracy: {stats['dssp']['min_accuracy']:.4f}\n")
                f.write(f"Max Accuracy: {stats['dssp']['max_accuracy']:.4f}\n")
                f.write(f"Mean F1-Score: {stats['dssp']['mean_f1']:.4f} ± {stats['dssp']['std_f1']:.4f}\n\n")
                
                cm = np.array(stats['dssp']['aggregate_confusion_matrix'])
                f.write("Aggregate Confusion Matrix (DSSP):\n")
                f.write(f"                Pred_Interior  Pred_Exterior\n")
                f.write(f"True_Interior   {cm[0,0]:14d}  {cm[0,1]:14d}\n")
                f.write(f"True_Exterior   {cm[1,0]:14d}  {cm[1,1]:14d}\n\n")
            
            # STRIDE results
            if 'stride' in stats:
                f.write("="*80 + "\n")
                f.write("RESULTS vs STRIDE\n")
                f.write("="*80 + "\n\n")
                f.write(f"Proteins with STRIDE data: {stats['stride']['n_proteins_with_data']}\n")
                f.write(f"Mean Accuracy: {stats['stride']['mean_accuracy']:.4f} ± {stats['stride']['std_accuracy']:.4f}\n")
                f.write(f"Median Accuracy: {stats['stride']['median_accuracy']:.4f}\n")
                f.write(f"Min Accuracy: {stats['stride']['min_accuracy']:.4f}\n")
                f.write(f"Max Accuracy: {stats['stride']['max_accuracy']:.4f}\n")
                f.write(f"Mean F1-Score: {stats['stride']['mean_f1']:.4f} ± {stats['stride']['std_f1']:.4f}\n\n")
                
                cm = np.array(stats['stride']['aggregate_confusion_matrix'])
                f.write("Aggregate Confusion Matrix (STRIDE):\n")
                f.write(f"                Pred_Interior  Pred_Exterior\n")
                f.write(f"True_Interior   {cm[0,0]:14d}  {cm[0,1]:14d}\n")
                f.write(f"True_Exterior   {cm[1,0]:14d}  {cm[1,1]:14d}\n\n")
            
            # Outlier analysis
            f.write("="*80 + "\n")
            f.write("OUTLIER ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            if 'dssp' in stats:
                dssp_accuracies = [(r.protein_id, r.dssp_accuracy) for r in results if r.dssp_accuracy is not None]
                dssp_accuracies.sort(key=lambda x: x[1])
                
                mean_acc = stats['dssp']['mean_accuracy']
                std_acc = stats['dssp']['std_accuracy']
                
                f.write("Low Performance Proteins (< mean - 1*std):\n")
                threshold_low = mean_acc - std_acc
                for pid, acc in dssp_accuracies:
                    if acc < threshold_low:
                        f.write(f"  {pid}: {acc:.4f}\n")
                
                f.write("\nHigh Performance Proteins (> mean + 1*std):\n")
                threshold_high = mean_acc + std_acc
                for pid, acc in reversed(dssp_accuracies):
                    if acc > threshold_high:
                        f.write(f"  {pid}: {acc:.4f}\n")
    
    def _save_per_protein_table(self, results: List[ProteinResults], filepath: Path):
        """Save per-protein accuracy table as CSV"""
        data = []
        for result in results:
            row = {
                'protein_id': result.protein_id,
                'n_residues': result.n_residues,
                'dssp_accuracy': result.dssp_accuracy,
                'dssp_precision': result.dssp_precision,
                'dssp_recall': result.dssp_recall,
                'dssp_f1': result.dssp_f1,
                'stride_accuracy': result.stride_accuracy,
                'stride_precision': result.stride_precision,
                'stride_recall': result.stride_recall,
                'stride_f1': result.stride_f1
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
    
    def identify_outliers(self, results: List[ProteinResults], reference: str = 'dssp') -> Dict:
        """
        Identify outlier proteins and analyze why they perform differently
        """
        if reference == 'dssp':
            accuracies = [(r.protein_id, r.dssp_accuracy, r.n_residues) 
                         for r in results if r.dssp_accuracy is not None]
        else:
            accuracies = [(r.protein_id, r.stride_accuracy, r.n_residues) 
                         for r in results if r.stride_accuracy is not None]
        
        if not accuracies:
            return {}
        
        pids, accs, sizes = zip(*accuracies)
        accs = np.array(accs)
        sizes = np.array(sizes)
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        
        outliers = {
            'low_performers': [],
            'high_performers': [],
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc)
        }
        
        for pid, acc, size in accuracies:
            if acc < mean_acc - std_acc:
                outliers['low_performers'].append({
                    'protein_id': pid,
                    'accuracy': float(acc),
                    'n_residues': int(size),
                    'deviation_from_mean': float(acc - mean_acc)
                })
            elif acc > mean_acc + std_acc:
                outliers['high_performers'].append({
                    'protein_id': pid,
                    'accuracy': float(acc),
                    'n_residues': int(size),
                    'deviation_from_mean': float(acc - mean_acc)
                })
        
        return outliers


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("DUDE DATASET COMPREHENSIVE ANALYSIS")
    print("="*80 + "\n")
    
    # Configuration
    pdb_directory = Path.cwd()  # Assumes PDB files are in current directory or subdirectories
    
    config = DUDEAnalysisConfig(
        n_folds=5,  # 5-fold cross-validation
        n_optimization_trials=100,
        optimization_reference='dssp'
    )
    
    # Initialize analyzer
    analyzer = DUDEDatasetAnalyzer(pdb_directory, config)
    
    if len(analyzer.pdb_files) == 0:
        print("ERROR: No PDB files found!")
        print("\nPlease place PDB files in one of these locations:")
        print("  1. Current directory")
        print("  2. Subdirectories (e.g., dude1/, dude2/)")
        return
    
    print(f"\nFound {len(analyzer.pdb_files)} proteins to analyze")
    print(f"Cross-validation: {config.n_folds}-fold")
    print(f"Optimization trials: {config.n_optimization_trials}\n")
    
    # Phase 1: Baseline analysis with default parameters
    print("="*80)
    print("PHASE 1: BASELINE ANALYSIS (Default Parameters)")
    print("="*80)
    
    default_params = BurialParameters()
    baseline_result = analyzer.run_analysis_with_params(default_params)
    analyzer.save_results(baseline_result, "baseline")
    
    # Phase 2: Cross-validation with default parameters
    print("\n" + "="*80)
    print("PHASE 2: CROSS-VALIDATION (Default Parameters)")
    print("="*80)
    
    cv_result = analyzer.cross_validate(default_params, config.n_folds)
    
    # Save CV results
    with open(analyzer.output_dir / "baseline" / "cross_validation_results.json", 'w') as f:
        json.dump(cv_result['cv_statistics'], f, indent=2)
    
    # Phase 3: Parameter optimization
    print("\n" + "="*80)
    print("PHASE 3: PARAMETER OPTIMIZATION")
    print("="*80)
    
    best_params = analyzer.optimize_parameters(
        n_trials=config.n_optimization_trials,
        reference=config.optimization_reference
    )
    
    # Phase 4: Final analysis with optimized parameters
    print("\n" + "="*80)
    print("PHASE 4: FINAL ANALYSIS (Optimized Parameters)")
    print("="*80)
    
    optimized_result = analyzer.run_analysis_with_params(best_params)
    analyzer.save_results(optimized_result, "optimized")
    
    # Phase 5: Outlier analysis
    print("\n" + "="*80)
    print("PHASE 5: OUTLIER ANALYSIS")
    print("="*80)
    
    outliers_dssp = analyzer.identify_outliers(optimized_result['results'], 'dssp')
    outliers_stride = analyzer.identify_outliers(optimized_result['results'], 'stride')
    
    # Save outliers
    with open(analyzer.output_dir / "optimized" / "outliers_analysis.json", 'w') as f:
        json.dump({'dssp': outliers_dssp, 'stride': outliers_stride}, f, indent=2)
    
    print(f"\nLow performers (DSSP): {len(outliers_dssp.get('low_performers', []))}")
    print(f"High performers (DSSP): {len(outliers_dssp.get('high_performers', []))}")
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {analyzer.output_dir}")
    print("\nGenerated outputs:")
    print("  baseline/ - Results with default parameters")
    print("  optimized/ - Results with optimized parameters")
    print("  optimization_trials.csv - All Optuna trials")
    print("  */confusion_matrices/ - 2 matrices per protein")
    print("  */protein_csvs/ - Detailed data per protein")
    print("  */summary_report.txt - Comprehensive statistics")
    print("  */per_protein_accuracy.csv - Accuracy table")
    print()


if __name__ == "__main__":
    main()

