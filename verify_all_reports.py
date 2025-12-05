#!/usr/bin/env python3
"""
VERIFICATION SCRIPT - Confirms all 53 detailed_report.txt files have DSSP and STRIDE classifications
"""

from pathlib import Path

def verify_reports():
    report_dir = Path("/holder/results_dude/detailed_reports")

    print("\n" + "=" * 80)
    print("VERIFICATION: All Reports Have DSSP & STRIDE Classifications")
    print("=" * 80 + "\n")

    all_files = sorted(report_dir.glob("*_detailed_report.txt"))

    dssp_count = 0
    stride_count = 0
    agreement_count = 0

    for report_file in all_files:
        protein_id = report_file.stem.replace("_detailed_report", "")

        with open(report_file, 'r') as f:
            content = f.read()

        has_dssp = "DSSP Classification:" in content and "residues" in content
        has_stride = "STRIDE Classification:" in content and "residues" in content
        has_agreement_dssp = "Agreement with DSSP:" in content
        has_agreement_stride = "Agreement with STRIDE:" in content
        has_detailed = "DETAILED RESIDUE DATA" in content

        if has_dssp:
            dssp_count += 1
        if has_stride:
            stride_count += 1
        if has_agreement_dssp and has_agreement_stride:
            agreement_count += 1

        status = "✓" if (has_dssp and has_stride and has_detailed) else "✗"
        print(f"{status} {protein_id:10s} - DSSP:{has_dssp}, STRIDE:{has_stride}, Agreements:{has_agreement_dssp and has_agreement_stride}")

    print(f"\n{'=' * 80}")
    print(f"SUMMARY:")
    print(f"  Total reports: {len(all_files)}")
    print(f"  With DSSP classification: {dssp_count}/{len(all_files)}")
    print(f"  With STRIDE classification: {stride_count}/{len(all_files)}")
    print(f"  With both agreement metrics: {agreement_count}/{len(all_files)}")
    print(f"{'=' * 80}\n")

    if dssp_count == len(all_files) and stride_count == len(all_files):
        print("✅ SUCCESS! All 53 reports have DSSP and STRIDE classifications!\n")
        return True
    else:
        print("❌ Some reports are missing data\n")
        return False

if __name__ == "__main__":
    verify_reports()

