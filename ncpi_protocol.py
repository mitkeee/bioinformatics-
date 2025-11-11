#!/usr/bin/env python3
"""
NCPI Protocol - Neighbor Count and Point Index Protocol
Comprehensive parameter-based burial classification with optimization

NCPI Parameters:
- nc6_threshold: Neighbor count threshold at 6Å radius
- nc10_threshold: Neighbor count threshold at 10Å radius
- uni6_threshold: Uniformity threshold at 6Å (homogeneous distribution)
- uni10_threshold: Uniformity threshold at 10Å (homogeneous distribution)
- dssp_cutoff: ASA cutoff for DSSP classification
- stride_cutoff: ASA cutoff for STRIDE classification
"""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import json

from comprehensive_burial_analysis import (
    BurialParameters,
    process_protein_dataset,
    save_confusion_matrices,
    generate_summary_report
)


@dataclass
class NCPIParameters:
    """NCPI Protocol Parameters"""
    # Core classification parameters
    nc6_threshold: float = 10.0      # Neighbor count at 6Å
    nc10_threshold: float = 18.0     # Neighbor count at 10Å
    uni6_threshold: float = 0.40     # Uniformity at 6Å
    uni10_threshold: float = 0.50    # Uniformity at 10Å

    # Reference method cutoffs
    dssp_asa_cutoff: float = 30.0    # DSSP ASA cutoff (Ų)
    stride_asa_cutoff: float = 24.0  # STRIDE ASA cutoff (Ų)

    # Additional parameters
    neighbor_radius_small: float = 6.0   # Small radius (Ų)
    neighbor_radius_large: float = 10.0  # Large radius (Ų)

    def to_burial_parameters(self) -> BurialParameters:
        """Convert to BurialParameters for compatibility"""
        return BurialParameters(
            nc6_threshold=self.nc6_threshold,
            nc10_threshold=self.nc10_threshold,
            uni6_threshold=self.uni6_threshold,
            uni10_threshold=self.uni10_threshold,
            dssp_asa_cutoff=self.dssp_asa_cutoff,
            stride_asa_cutoff=self.stride_asa_cutoff
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'NCPIParameters':
        """Create from dictionary"""
        return cls(**d)

    def save(self, filepath: Path):
        """Save parameters to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"NCPI parameters saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'NCPIParameters':
        """Load parameters from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class NCPIProtocol:
    """
    NCPI Protocol - Neighbor Count and Point Index Protocol
    Main interface for running burial classification with parameter optimization
    """

    def __init__(self, workspace_dir: Path = None):
        self.workspace_dir = workspace_dir or Path.cwd()
        self.output_dir = self.workspace_dir / "results" / "ncpi_protocol"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Default parameters
        self.default_params = NCPIParameters()
        self.current_params = self.default_params

        # Find PDB files
        self.pdb_files = sorted(self.workspace_dir.glob("*.pdb"))

    def run_with_parameters(self, params: NCPIParameters, save_results: bool = True) -> Dict:
        """
        Run NCPI protocol with given parameters
        Returns dictionary with results and statistics
        """
        print("\n" + "="*80)
        print("NCPI PROTOCOL - RUNNING ANALYSIS")
        print("="*80)
        print(f"\nParameters:")
        for key, value in params.to_dict().items():
            print(f"  {key}: {value}")
        print()

        # Convert to BurialParameters
        burial_params = params.to_burial_parameters()

        # Process proteins
        results = process_protein_dataset(self.pdb_files, burial_params)

        # Calculate statistics
        stats = self._calculate_statistics(results)

        # Save if requested
        if save_results:
            # Create parameter-specific subdirectory
            param_str = f"nc6_{params.nc6_threshold:.1f}_nc10_{params.nc10_threshold:.1f}"
            param_dir = self.output_dir / param_str
            param_dir.mkdir(exist_ok=True)

            # Save confusion matrices
            save_confusion_matrices(results, param_dir / "confusion_matrices")

            # Save summary report
            generate_summary_report(results, param_dir / "summary_report.txt")

            # Save parameters
            params.save(param_dir / "ncpi_parameters.json")

            # Save statistics
            self._save_statistics(stats, param_dir / "statistics.json")

            print(f"\nResults saved to: {param_dir}")

        return {
            'parameters': params,
            'results': results,
            'statistics': stats
        }

    def parameter_sweep(self,
                       nc6_range: List[float],
                       nc10_range: List[float],
                       uni6_range: List[float],
                       uni10_range: List[float]) -> pd.DataFrame:
        """
        Perform parameter sweep to find optimal parameters
        Tests all combinations and returns results DataFrame
        """
        print("\n" + "="*80)
        print("NCPI PROTOCOL - PARAMETER SWEEP")
        print("="*80)
        print(f"\nParameter ranges:")
        print(f"  nc6_threshold: {nc6_range}")
        print(f"  nc10_threshold: {nc10_range}")
        print(f"  uni6_threshold: {uni6_range}")
        print(f"  uni10_threshold: {uni10_range}")
        print()

        results_list = []
        total_combinations = len(nc6_range) * len(nc10_range) * len(uni6_range) * len(uni10_range)
        count = 0

        for nc6 in nc6_range:
            for nc10 in nc10_range:
                for uni6 in uni6_range:
                    for uni10 in uni10_range:
                        count += 1
                        print(f"Testing combination {count}/{total_combinations}...", end='\r')

                        # Create parameters
                        params = NCPIParameters(
                            nc6_threshold=nc6,
                            nc10_threshold=nc10,
                            uni6_threshold=uni6,
                            uni10_threshold=uni10
                        )

                        # Run analysis (without saving)
                        result = self.run_with_parameters(params, save_results=False)

                        # Extract statistics
                        stats = result['statistics']

                        # Add to results
                        result_dict = {
                            'nc6_threshold': nc6,
                            'nc10_threshold': nc10,
                            'uni6_threshold': uni6,
                            'uni10_threshold': uni10,
                            'dssp_mean_accuracy': stats['dssp']['mean_accuracy'],
                            'dssp_std_accuracy': stats['dssp']['std_accuracy'],
                            'dssp_mean_f1': stats['dssp']['mean_f1'],
                            'stride_mean_accuracy': stats['stride']['mean_accuracy'],
                            'stride_std_accuracy': stats['stride']['std_accuracy'],
                            'stride_mean_f1': stats['stride']['mean_f1']
                        }
                        results_list.append(result_dict)

        print(f"\nCompleted {count} parameter combinations")

        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)

        # Sort by accuracy
        results_df = results_df.sort_values('dssp_mean_accuracy', ascending=False)

        # Save results
        results_df.to_csv(self.output_dir / "parameter_sweep_results.csv", index=False)
        print(f"Parameter sweep results saved to: {self.output_dir / 'parameter_sweep_results.csv'}")

        # Display top 10 results
        print("\n" + "="*80)
        print("TOP 10 PARAMETER COMBINATIONS")
        print("="*80)
        print(results_df.head(10).to_string())

        return results_df

    def run_default(self):
        """Run with default parameters"""
        return self.run_with_parameters(self.default_params)

    def _calculate_statistics(self, results) -> Dict:
        """Calculate statistics from results"""
        stats = {
            'n_proteins': len(results),
            'total_residues': sum(r.n_residues for r in results),
            'dssp': {},
            'stride': {}
        }

        # DSSP statistics
        dssp_accuracies = [r.dssp_accuracy for r in results if r.dssp_accuracy is not None]
        dssp_f1_scores = [r.dssp_f1 for r in results if r.dssp_f1 is not None]

        if dssp_accuracies:
            stats['dssp'] = {
                'mean_accuracy': float(np.mean(dssp_accuracies)),
                'std_accuracy': float(np.std(dssp_accuracies)),
                'min_accuracy': float(np.min(dssp_accuracies)),
                'max_accuracy': float(np.max(dssp_accuracies)),
                'mean_f1': float(np.mean(dssp_f1_scores)),
                'std_f1': float(np.std(dssp_f1_scores))
            }

        # STRIDE statistics
        stride_accuracies = [r.stride_accuracy for r in results if r.stride_accuracy is not None]
        stride_f1_scores = [r.stride_f1 for r in results if r.stride_f1 is not None]

        if stride_accuracies:
            stats['stride'] = {
                'mean_accuracy': float(np.mean(stride_accuracies)),
                'std_accuracy': float(np.std(stride_accuracies)),
                'min_accuracy': float(np.min(stride_accuracies)),
                'max_accuracy': float(np.max(stride_accuracies)),
                'mean_f1': float(np.mean(stride_f1_scores)),
                'std_f1': float(np.std(stride_f1_scores))
            }

        return stats

    def _save_statistics(self, stats: Dict, filepath: Path):
        """Save statistics to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("NCPI PROTOCOL - Neighbor Count and Point Index Protocol")
    print("="*80)
    print()

    # Initialize protocol
    protocol = NCPIProtocol()

    print(f"Found {len(protocol.pdb_files)} PDB files:")
    for pdb in protocol.pdb_files:
        print(f"  - {pdb.name}")
    print()

    # Option 1: Run with default parameters
    print("Option 1: Run with default parameters")
    print("  python3 ncpi_protocol.py --default")
    print()

    # Option 2: Run parameter sweep
    print("Option 2: Run parameter sweep (tests multiple combinations)")
    print("  python3 ncpi_protocol.py --sweep")
    print()

    # Run default analysis
    print("Running with default parameters...")
    result = protocol.run_default()

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    stats = result['statistics']

    if 'dssp' in stats and stats['dssp']:
        print(f"\nDSSP Comparison:")
        print(f"  Mean Accuracy: {stats['dssp']['mean_accuracy']:.4f} ± {stats['dssp']['std_accuracy']:.4f}")
        print(f"  Mean F1-Score: {stats['dssp']['mean_f1']:.4f}")
        print(f"  Accuracy Range: [{stats['dssp']['min_accuracy']:.4f}, {stats['dssp']['max_accuracy']:.4f}]")

    if 'stride' in stats and stats['stride']:
        print(f"\nSTRIDE Comparison:")
        print(f"  Mean Accuracy: {stats['stride']['mean_accuracy']:.4f} ± {stats['stride']['std_accuracy']:.4f}")
        print(f"  Mean F1-Score: {stats['stride']['mean_f1']:.4f}")
        print(f"  Accuracy Range: [{stats['stride']['min_accuracy']:.4f}, {stats['stride']['max_accuracy']:.4f}]")

    print("\n" + "="*80)
    print("✓ NCPI PROTOCOL COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {protocol.output_dir}")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--sweep":
            # Run parameter sweep
            protocol = NCPIProtocol()
            protocol.parameter_sweep(
                nc6_range=[8.0, 9.0, 10.0, 11.0, 12.0],
                nc10_range=[16.0, 18.0, 20.0, 22.0, 24.0],
                uni6_range=[0.35, 0.40, 0.45, 0.50],
                uni10_range=[0.45, 0.50, 0.55, 0.60]
            )
        elif sys.argv[1] == "--default":
            main()
        else:
            print("Usage: python3 ncpi_protocol.py [--default|--sweep]")
    else:
        main()

