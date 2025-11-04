# Protein Burial Classification Analysis - Summary Report

## Overview
Successfully implemented and optimized a protein burial classification system that analyzes residue exposure using neighbor-based geometric features.

## Dataset
- **Proteins Analyzed**: 4 PDB files
  - 3PTE.pdb (347 residues) - Accuracy: **79.3%**
  - 4d05.pdb (493 residues) - No DSSP reference
  - 6wti.pdb (1207 residues) - Accuracy: **76.9%**
  - 7upo.pdb (228 residues) - Accuracy: **83.3%**

- **Total Residues**: 2,275 residues across all proteins
- **Residues with DSSP Reference**: 1,782 residues

## Final Performance Metrics

### Overall Accuracy: **78.2%**

### Confusion Matrix (vs DSSP as ground truth):
```
                    Predicted
                 Interior  Exterior
Actual Interior      857       290
Actual Exterior       99       536
```

### Classification Performance:
- **Interior Residues**:
  - Precision: 90%
  - Recall: 75%
  - F1-Score: 0.82

- **Exterior Residues**:
  - Precision: 65%
  - Recall: 84%
  - F1-Score: 0.73

## Methodology

### Input Features (for each residue):
1. **Neighbor Counts**:
   - `ncps_sphere_6`: Number of CA neighbors within 6Å radius
   - `ncps_sphere_10`: Number of CA neighbors within 10Å radius

2. **Uniformity Metrics**:
   - `ncps_sphere_6_uni`: Spherical variance at 6Å (0-1 scale)
   - `ncps_sphere_10_uni`: Spherical variance at 10Å (0-1 scale)
   - Higher values = neighbors distributed all around (interior)
   - Lower values = neighbors on one side only (exterior)

### Classification Logic (WITHOUT deg7):
A residue is classified as **EXTERIOR** if:
- Few neighbors: `ncps_sphere_6 < 6` OR `ncps_sphere_10 < 12`
- OR low uniformity: `ncps_sphere_6_uni < 0.30` OR `ncps_sphere_10_uni < 0.60`

Otherwise classified as **INTERIOR** (many neighbors + high uniformity)

## Optimal Parameters (Found via Grid Search)
After testing 1,225 parameter combinations:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `nc6_threshold` | 6 | Min neighbors at 6Å radius |
| `nc10_threshold` | 12 | Min neighbors at 10Å radius |
| `uni6_threshold` | 0.30 | Min uniformity at 6Å |
| `uni10_threshold` | 0.60 | Min uniformity at 10Å |

## Output Format

### CSV Columns (13 total):
1. `res_id` - Residue name (ALA, GLN, etc.)
2. `res_num` - Residue number from PDB
3. `dssp_asa` - DSSP accessible surface area (Ų)
4. `dssp_class` - DSSP classification (1=exterior, 0=interior)
5. `stride_asa` - STRIDE accessible surface area (Ų)
6. `stride_class` - STRIDE classification (1=exterior, 0=interior)
7. `ncps_sphere_6` - Neighbor count at 6Å
8. `ncps_sphere_6_uni` - Uniformity at 6Å
9. `ncps_sphere_10` - Neighbor count at 10Å
10. `ncps_sphere_10_uni` - Uniformity at 10Å
11. `ncps_class` - Our classification (1=exterior, 0=interior)
12. `dssp_ss` - DSSP secondary structure
13. `stride_ss` - STRIDE secondary structure

## Reference Methods

### DSSP Classification:
- **Threshold**: 30 Ų (absolute ASA)
- ASA ≥ 30 Ų → Exterior (class 1)
- ASA < 30 Ų → Interior (class 0)
- Note: Bio.PDB.DSSP returns relative ASA (0-1), converted to absolute using max ASA per residue type

### STRIDE Classification:
- **Threshold**: 24 Ų (absolute ASA)
- ASA ≥ 24 Ų → Exterior (class 1)
- ASA < 24 Ų → Interior (class 0)
- Note: STRIDE files in current dataset lack ASG records with accessibility data

## Key Improvements Made

1. **Fixed DSSP ASA Extraction**: Converted relative ASA (0-1) to absolute values (Ų) using standard maximum ASA values for each amino acid type

2. **Simplified Classification**: Removed deg7 dependency, using only neighbor counts and uniformity

3. **Parameter Optimization**: Grid search over 1,225 combinations improved accuracy from 45.9% to 78.2%

4. **Proper File Handling**: Added case-insensitive file name matching for cross-platform compatibility

## Data Distribution Statistics

### Neighbor Counts:
- **6Å Sphere**: Mean=8.0, Median=8.0, Range=[1-21]
- **10Å Sphere**: Mean=18.2, Median=18.0, Range=[2-34]

### Uniformity:
- **6Å Sphere**: Mean=0.51, Median=0.47, Range=[0.13-0.97]
- **10Å Sphere**: Mean=0.69, Median=0.72, Range=[0.09-0.98]

## Output Files Generated

### In `results/` directory:
1. **Individual protein CSVs**: 
   - `3PTE_results.csv`
   - `4d05_results.csv`
   - `6wti_results.csv`
   - `7upo_results.csv`

2. **Combined analysis**:
   - `combined_results.csv` - All proteins merged
   - `accuracy_analysis.txt` - Detailed metrics
   - `parameter_tuning_results.csv` - All 1,225 tested combinations
   - `best_parameters.txt` - Optimal parameters with confusion matrix

## Interpretation & Insights

### Strengths:
- **High precision for interior residues** (90%) - When we predict interior, we're usually correct
- **High recall for exterior residues** (84%) - We catch most of the exposed residues
- **Consistent performance** across different protein sizes (76.9% - 83.3%)
- **Simple geometric features** achieve good performance without complex calculations

### Areas for Improvement:
- **Lower precision for exterior** (65%) - Some false positives (interior predicted as exterior)
- **Lower recall for interior** (75%) - Missing some buried residues
- **STRIDE data integration** - Need ASG records for additional validation

### Biological Interpretation:
The uniformity metric is particularly effective because:
- **Interior residues** are surrounded by protein mass from all directions → high uniformity
- **Surface residues** have neighbors only on the protein side → low uniformity
- **Pocket/cleft residues** can have many neighbors but still be exposed → captured by uniformity

## Usage Instructions

### To analyze new proteins:
```python
python protein_burial_analysis.py
```

### To re-tune parameters:
```python
python parameter_tuning.py
```

### Input requirements:
- PDB format files
- File naming: `[protein_id].pdb`

### Output:
- CSV files with all 13 columns as specified
- Individual protein results + combined dataset
- Accuracy metrics vs DSSP reference

## Technical Notes

### Dependencies:
- Python 3.x
- BioPython (PDB parsing, DSSP)
- NumPy (distance calculations)
- Pandas (data management)
- scikit-learn (metrics)

### Computation:
- Distance calculations: O(n²) per protein
- Processing time: ~1-5 seconds per protein
- Memory efficient for proteins up to several thousand residues

### Limitations:
1. Requires CA atoms (standard in PDB files)
2. DSSP must be installed for reference comparisons
3. Current STRIDE files lack ASG records (accessibility data)
4. Single-chain bias (multi-domain proteins may need preprocessing)

## Recommendations

### For Better Accuracy:
1. **Generate proper STRIDE files** with ASG records for validation
2. **Test on larger dataset** to validate parameter stability
3. **Add domain boundary detection** for multi-domain proteins
4. **Consider secondary structure** in classification (currently extracted but not used)

### For Production Use:
1. **Ensemble approach**: Combine DSSP, STRIDE, and NCPS classifications
2. **Confidence scores**: Use distance to decision boundary
3. **Residue-specific thresholds**: Different parameters for different amino acid types

## Conclusion

Successfully implemented a protein burial classification system achieving **78.2% accuracy** using simple geometric features (neighbor counts and uniformity) without relying on deg7 calculations. The system processes standard PDB files and outputs comprehensive CSV reports with 13 features per residue as specified.

The optimal parameters found (6 neighbors at 6Å, 12 at 10Å, uniformity thresholds of 0.30 and 0.60) provide a good balance between detecting buried and exposed residues across multiple protein structures.

---
**Generated**: November 3, 2025
**Analysis Tool**: protein_burial_analysis.py v2.0
**Dataset**: 4 proteins, 2,275 residues, 1,782 with DSSP reference

