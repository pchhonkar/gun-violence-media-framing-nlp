"""
End-to-end pipeline runner for HW5 NLP.

Executes all pipeline steps in order:
1. load_articles        → articles_master.csv
2. coref_contexts       → contexts_victims.jsonl, contexts_shooters.jsonl
3. extract_descriptions → descriptions.csv, descriptions_raw.csv
4. embed_cluster        → descriptions_with_clusters.csv, cluster_summary.csv
5. manual_eval_helpers  → cluster_refinement_map.json, descriptions_with_clusters_refined.csv
6. task5_frequency_analysis → frequency/proportion tables
7. task6_chi_square     → chi_square_results.csv

Usage:
    python -m src.run_all
    python -m src.run_all --force        # Re-run all steps even if outputs exist
    python -m src.run_all --skip-task4   # Skip manual refinement step

Saves logs to outputs/reports/run_log.txt
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

# Import config for paths
try:
    from . import config
except ImportError:
    import config


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

PIPELINE_STEPS = [
    {
        "name": "Task 1a: Load Articles",
        "module": "src.load_articles",
        "expected_outputs": [
            "outputs/processed/articles_master.csv",
        ],
        "skip_flag": None,
    },
    {
        "name": "Task 1b: Coreference & Context Extraction",
        "module": "src.coref_contexts",
        "expected_outputs": [
            "outputs/processed/contexts_victims.jsonl",
            "outputs/processed/contexts_shooters.jsonl",
            "outputs/reports/task1_doc.md",
        ],
        "skip_flag": None,
    },
    {
        "name": "Task 2: Extract Descriptions",
        "module": "src.extract_descriptions",
        "expected_outputs": [
            "outputs/processed/descriptions.csv",
            "outputs/processed/descriptions_raw.csv",
            "outputs/reports/task2_rationale.md",
        ],
        "skip_flag": None,
    },
    {
        "name": "Task 3: Embedding & Clustering",
        "module": "src.embed_cluster",
        "expected_outputs": [
            "outputs/processed/descriptions_with_clusters.csv",
            "outputs/processed/cluster_summary.csv",
            "outputs/figures/umap_clusters_victim.png",
            "outputs/figures/umap_clusters_shooter.png",
            "outputs/reports/task3_doc.md",
            "outputs/reports/dbscan_tuning.md",
        ],
        "skip_flag": None,
    },
    {
        "name": "Task 4: Manual Evaluation & Refinement",
        "module": "src.manual_eval_helpers",
        "expected_outputs": [
            "outputs/reports/task4_manual_eval.md",
            "outputs/processed/cluster_refinement_map.json",
            "outputs/processed/descriptions_with_clusters_refined.csv",
        ],
        "skip_flag": "skip_task4",
    },
    {
        "name": "Task 5: Frequency Analysis",
        "module": "src.task5_frequency_analysis",
        "expected_outputs": [
            "outputs/processed/frequency_table_victim.csv",
            "outputs/processed/frequency_table_shooter.csv",
            "outputs/processed/proportion_table_victim.csv",
            "outputs/processed/proportion_table_shooter.csv",
            "outputs/figures/task5_heatmap_victim.png",
            "outputs/figures/task5_heatmap_shooter.png",
            "outputs/figures/task5_bar_top6_victim.png",
            "outputs/figures/task5_bar_top6_shooter.png",
            "outputs/reports/task5_doc.md",
        ],
        "skip_flag": None,
    },
    {
        "name": "Task 6: Chi-Square Hypothesis Testing",
        "module": "src.task6_chi_square",
        "expected_outputs": [
            "outputs/processed/task6_chi_square_results.csv",
            "outputs/reports/task6_results.md",
        ],
        "skip_flag": None,
    },
]


# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    """Logger that writes to both console and file."""
    
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer = []
    
    def log(self, message: str, newline: bool = True):
        """Log a message to console and buffer."""
        if newline:
            print(message)
            self.buffer.append(message + "\n")
        else:
            print(message, end="")
            self.buffer.append(message)
    
    def timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def save(self):
        """Save log buffer to file."""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.writelines(self.buffer)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_outputs_exist(expected_outputs: List[str], project_root: Path) -> Tuple[bool, List[str]]:
    """
    Check if expected output files exist.
    
    Returns:
        Tuple of (all_exist, missing_files)
    """
    missing = []
    for output in expected_outputs:
        path = project_root / output
        if not path.exists():
            missing.append(output)
    
    return len(missing) == 0, missing


def run_module(module: str, logger: Logger) -> Tuple[bool, str]:
    """
    Run a Python module using subprocess.
    
    Args:
        module: Module name (e.g., "src.load_articles")
        logger: Logger instance
    
    Returns:
        Tuple of (success, output/error message)
    """
    cmd = [sys.executable, "-m", module]
    logger.log(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env={**dict(__import__('os').environ), 'PYTHONUNBUFFERED': '1'}
        )
        return True, result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        error_msg = f"Exit code: {e.returncode}\n"
        error_msg += f"STDOUT:\n{e.stdout}\n" if e.stdout else ""
        error_msg += f"STDERR:\n{e.stderr}\n" if e.stderr else ""
        return False, error_msg
    except Exception as e:
        return False, f"Exception: {type(e).__name__}: {e}"


# =============================================================================
# MAIN PIPELINE RUNNER
# =============================================================================

def run_pipeline(force: bool = False, skip_task4: bool = False) -> int:
    """
    Run the complete NLP pipeline.
    
    Args:
        force: Re-run steps even if outputs exist
        skip_task4: Skip the manual evaluation step
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Setup
    project_root = Path(__file__).parent.parent
    log_path = project_root / "outputs" / "reports" / "run_log.txt"
    logger = Logger(log_path)
    
    # Ensure output directories exist
    config.ensure_output_dirs()
    
    # Header
    logger.log("=" * 70)
    logger.log("HW5 NLP PIPELINE - FULL EXECUTION")
    logger.log("=" * 70)
    logger.log(f"Started at: {logger.timestamp()}")
    logger.log(f"Python: {sys.executable}")
    logger.log(f"Version: {sys.version.split()[0]}")
    logger.log(f"Working directory: {project_root}")
    logger.log(f"Options: force={force}, skip_task4={skip_task4}")
    logger.log("=" * 70)
    
    pipeline_start = time.time()
    
    # Track results
    results: List[Tuple[str, str, float]] = []  # (step_name, status, duration)
    
    # Run each step
    for step_config in PIPELINE_STEPS:
        step_name = step_config["name"]
        module = step_config["module"]
        expected_outputs = step_config["expected_outputs"]
        skip_flag = step_config["skip_flag"]
        
        logger.log("")
        logger.log("-" * 70)
        logger.log(f"STEP: {step_name}")
        logger.log(f"Module: {module}")
        logger.log(f"Timestamp: {logger.timestamp()}")
        logger.log("-" * 70)
        
        step_start = time.time()
        
        # Check if step should be skipped via flag
        if skip_flag == "skip_task4" and skip_task4:
            duration = time.time() - step_start
            logger.log(f"  STATUS: SKIPPED (--skip-task4 flag)")
            results.append((step_name, "SKIPPED (flag)", duration))
            continue
        
        # Check if outputs already exist (skip unless --force)
        if not force:
            all_exist, missing = check_outputs_exist(expected_outputs, project_root)
            if all_exist:
                duration = time.time() - step_start
                logger.log(f"  STATUS: SKIPPED (outputs already exist)")
                logger.log(f"  Existing files: {len(expected_outputs)}")
                results.append((step_name, "SKIPPED (exists)", duration))
                continue
        
        # Run the module
        logger.log(f"  Running...")
        success, output = run_module(module, logger)
        duration = time.time() - step_start
        
        if success:
            logger.log(f"  STATUS: SUCCESS ({duration:.1f}s)")
            
            # Verify outputs were created
            all_exist, missing = check_outputs_exist(expected_outputs, project_root)
            if all_exist:
                logger.log(f"  Verified: {len(expected_outputs)} output files created")
            else:
                logger.log(f"  WARNING: Missing outputs: {missing}")
            
            results.append((step_name, "SUCCESS", duration))
        else:
            logger.log(f"  STATUS: FAILED ({duration:.1f}s)")
            logger.log(f"  Error output:")
            for line in output.split('\n')[:20]:  # Limit error output
                logger.log(f"    {line}")
            
            results.append((step_name, "FAILED", duration))
            
            # Stop pipeline on failure
            logger.log("")
            logger.log("=" * 70)
            logger.log(f"⚠ PIPELINE STOPPED: {step_name} failed")
            logger.log("=" * 70)
            break
    
    # Pipeline summary
    pipeline_duration = time.time() - pipeline_start
    
    logger.log("")
    logger.log("=" * 70)
    logger.log("PIPELINE SUMMARY")
    logger.log("=" * 70)
    logger.log(f"Finished at: {logger.timestamp()}")
    logger.log(f"Total duration: {pipeline_duration:.1f}s ({pipeline_duration/60:.1f} min)")
    logger.log("")
    logger.log(f"{'Step':<45} {'Status':<20} {'Time':<10}")
    logger.log("-" * 70)
    
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for step_name, status, duration in results:
        short_name = step_name[:43] if len(step_name) > 43 else step_name
        logger.log(f"{short_name:<45} {status:<20} {duration:.1f}s")
        
        if "SUCCESS" in status:
            success_count += 1
        elif "SKIPPED" in status:
            skipped_count += 1
        else:
            failed_count += 1
    
    logger.log("-" * 70)
    logger.log(f"Total: {success_count} succeeded, {skipped_count} skipped, {failed_count} failed")
    logger.log("")
    
    # Final status
    if failed_count == 0:
        logger.log("=" * 70)
        logger.log("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.log("=" * 70)
        logger.log("")
        logger.log("Output locations:")
        logger.log("  Processed data: outputs/processed/")
        logger.log("  Figures:        outputs/figures/")
        logger.log("  Reports:        outputs/reports/")
        exit_code = 0
    else:
        logger.log("=" * 70)
        logger.log("✗ PIPELINE FAILED")
        logger.log("=" * 70)
        exit_code = 1
    
    # Save log
    logger.save()
    print(f"\nLog saved to: {log_path}")
    
    return exit_code


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the complete HW5 NLP pipeline end-to-end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.run_all              # Run pipeline, skip if outputs exist
  python -m src.run_all --force      # Re-run all steps
  python -m src.run_all --skip-task4 # Skip manual refinement step
        """
    )
    
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-run steps even if output files already exist"
    )
    
    parser.add_argument(
        "--skip-task4",
        action="store_true",
        help="Skip Task 4 (manual evaluation and refinement)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    exit_code = run_pipeline(force=args.force, skip_task4=args.skip_task4)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
