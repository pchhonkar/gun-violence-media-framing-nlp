"""
End-to-end pipeline runner for HW5 NLP.

Executes all pipeline steps in order:
  Task 1a: load_articles
  Task 1b: coref_contexts
  Task 2:  extract_descriptions
  Task 3:  embed_cluster
  Task 4:  manual_eval_helpers (skippable)
  Task 5:  task5_frequency_analysis
  Task 6:  task6_chi_square

Usage:
    python -m src.run_all
    python -m src.run_all --force        # Re-run all steps even if outputs exist
    python -m src.run_all --skip-task4   # Skip manual refinement step

Saves timestamped log to: outputs/reports/run_all_log_YYYYMMDD_HHMMSS.txt
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Determine project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Import config for paths
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from src import config
except ImportError:
    import config


# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

def get_pipeline_steps() -> List[Dict[str, Any]]:
    """
    Get pipeline step configurations using config paths.
    Returns list of step dictionaries with Path objects for expected outputs.
    """
    return [
        {
            "name": "Task 1a: Load Articles",
            "module": "src.load_articles",
            "expected_outputs": [
                config.PROCESSED_DIR / "articles_master.csv",
            ],
            "skip_flag": None,
        },
        {
            "name": "Task 1b: Coreference & Context Extraction",
            "module": "src.coref_contexts",
            "expected_outputs": [
                config.PROCESSED_DIR / "contexts_victims.jsonl",
                config.PROCESSED_DIR / "contexts_shooters.jsonl",
                config.REPORTS_DIR / "task1_doc.md",
            ],
            "skip_flag": None,
        },
        {
            "name": "Task 2: Extract Descriptions",
            "module": "src.extract_descriptions",
            "expected_outputs": [
                config.PROCESSED_DIR / "descriptions.csv",
                config.PROCESSED_DIR / "descriptions_raw.csv",
                config.REPORTS_DIR / "task2_rationale.md",
            ],
            "skip_flag": None,
        },
        {
            "name": "Task 3: Embedding & Clustering",
            "module": "src.embed_cluster",
            "expected_outputs": [
                config.PROCESSED_DIR / "descriptions_with_clusters.csv",
                config.PROCESSED_DIR / "cluster_summary.csv",
                config.FIGURES_DIR / "umap_clusters_victim.png",
                config.FIGURES_DIR / "umap_clusters_shooter.png",
                config.REPORTS_DIR / "task3_doc.md",
                config.REPORTS_DIR / "dbscan_tuning.md",
            ],
            "skip_flag": None,
        },
        {
            "name": "Task 4: Manual Evaluation & Refinement",
            "module": "src.manual_eval_helpers",
            "expected_outputs": [
                config.REPORTS_DIR / "task4_manual_eval.md",
                config.PROCESSED_DIR / "cluster_refinement_map.json",
                config.PROCESSED_DIR / "descriptions_with_clusters_refined.csv",
            ],
            "skip_flag": "skip_task4",
        },
        {
            "name": "Task 5: Frequency Analysis",
            "module": "src.task5_frequency_analysis",
            "expected_outputs": [
                config.PROCESSED_DIR / "frequency_table_victim.csv",
                config.PROCESSED_DIR / "frequency_table_shooter.csv",
                config.PROCESSED_DIR / "proportion_table_victim.csv",
                config.PROCESSED_DIR / "proportion_table_shooter.csv",
                config.FIGURES_DIR / "task5_heatmap_victim.png",
                config.FIGURES_DIR / "task5_heatmap_shooter.png",
                config.FIGURES_DIR / "task5_bar_top6_victim.png",
                config.FIGURES_DIR / "task5_bar_top6_shooter.png",
                config.REPORTS_DIR / "task5_doc.md",
            ],
            "skip_flag": None,
        },
        {
            "name": "Task 6: Chi-Square Hypothesis Testing",
            "module": "src.task6_chi_square",
            "expected_outputs": [
                config.PROCESSED_DIR / "task6_chi_square_results.csv",
                config.REPORTS_DIR / "task6_results.md",
            ],
            "skip_flag": None,
        },
    ]


# =============================================================================
# LOGGING
# =============================================================================

class PipelineLogger:
    """Logger that writes to both console (short) and file (full)."""
    
    def __init__(self):
        # Ensure output directories exist BEFORE using config paths
        config.ensure_output_dirs()
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"run_all_log_{timestamp}.txt"
        self.log_path = config.REPORTS_DIR / self.log_filename
        
        # Initialize log buffer
        self.log_buffer: List[str] = []
        self._write_header()
    
    def _write_header(self):
        """Write log file header."""
        self._log_to_file("=" * 80)
        self._log_to_file("HW5 NLP PIPELINE - EXECUTION LOG")
        self._log_to_file("=" * 80)
        self._log_to_file(f"Log file: {self.log_path}")
        self._log_to_file(f"Created: {self.timestamp()}")
        self._log_to_file(f"Project root: {PROJECT_ROOT}")
        self._log_to_file(f"Current working directory: {Path.cwd()}")
        self._log_to_file(f"Python executable: {sys.executable}")
        self._log_to_file(f"Python version: {sys.version}")
        self._log_to_file("=" * 80)
        self._log_to_file("")
    
    def timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _log_to_file(self, message: str):
        """Add message to log buffer."""
        self.log_buffer.append(message + "\n")
    
    def console(self, message: str):
        """Print to console only."""
        print(message)
    
    def log(self, message: str, console: bool = True):
        """Log to file and optionally to console."""
        self._log_to_file(message)
        if console:
            print(message)
    
    def log_step_start(self, step_name: str, module: str, command: List[str]):
        """Log step start details."""
        self._log_to_file("")
        self._log_to_file("-" * 80)
        self._log_to_file(f"STEP: {step_name}")
        self._log_to_file("-" * 80)
        self._log_to_file(f"Module: {module}")
        self._log_to_file(f"Command: {' '.join(command)}")
        self._log_to_file(f"Working directory: {PROJECT_ROOT}")
        self._log_to_file(f"Start time: {self.timestamp()}")
        self._log_to_file("")
    
    def log_step_output(self, stdout: str, stderr: str):
        """Log full stdout and stderr (NOT truncated)."""
        if stdout:
            self._log_to_file("--- STDOUT ---")
            self._log_to_file(stdout)
        if stderr:
            self._log_to_file("--- STDERR ---")
            self._log_to_file(stderr)
    
    def log_step_end(self, status: str, duration: float, end_time: str):
        """Log step completion details."""
        self._log_to_file("")
        self._log_to_file(f"End time: {end_time}")
        self._log_to_file(f"Duration: {duration:.2f} seconds")
        self._log_to_file(f"Status: {status}")
        self._log_to_file("-" * 80)
    
    def log_missing_outputs(self, missing: List[Path]):
        """Log missing output files."""
        if missing:
            self._log_to_file("")
            self._log_to_file("WARNING: Missing expected outputs:")
            for path in missing:
                self._log_to_file(f"  - {path}")
    
    def save(self):
        """Save log buffer to file."""
        with open(self.log_path, 'w', encoding='utf-8') as f:
            f.writelines(self.log_buffer)
        return self.log_path


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_outputs_exist(expected_outputs: List[Path]) -> Tuple[bool, List[Path]]:
    """
    Check if expected output files exist.
    
    Args:
        expected_outputs: List of Path objects for expected files
    
    Returns:
        Tuple of (all_exist, missing_files)
    """
    missing: List[Path] = []
    for path in expected_outputs:
        if not path.exists():
            missing.append(path)
    return len(missing) == 0, missing


def run_module(step_name: str, module: str, logger: PipelineLogger) -> Tuple[bool, str, str]:
    """
    Run a Python module using subprocess from PROJECT_ROOT.
    
    Args:
        step_name: Human-readable step name
        module: Module name (e.g., "src.load_articles")
        logger: Logger instance
    
    Returns:
        Tuple of (success, stdout, stderr)
    """
    cmd = [sys.executable, "-m", module]
    logger.log_step_start(step_name, module, cmd)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            cwd=PROJECT_ROOT  # Always run from project root
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout or "", e.stderr or ""
    except Exception as e:
        return False, "", f"Exception: {type(e).__name__}: {e}"


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
    # Setup logger (also ensures output dirs exist)
    logger = PipelineLogger()
    
    # Console header
    logger.console("")
    logger.console("=" * 70)
    logger.console("HW5 NLP PIPELINE - FULL EXECUTION")
    logger.console("=" * 70)
    logger.console(f"Started: {logger.timestamp()}")
    logger.console(f"Project root: {PROJECT_ROOT}")
    logger.console(f"Options: force={force}, skip_task4={skip_task4}")
    logger.console(f"Log file: {logger.log_path}")
    logger.console("=" * 70)
    
    # Log options
    logger._log_to_file(f"Options: force={force}, skip_task4={skip_task4}")
    logger._log_to_file("")
    
    pipeline_start = time.time()
    
    # Get pipeline steps
    pipeline_steps = get_pipeline_steps()
    
    # Track results: (step_name, status, duration, missing_outputs)
    results: List[Tuple[str, str, float, List[Path]]] = []
    all_missing_outputs: List[Path] = []
    
    # Run each step
    for step_config in pipeline_steps:
        step_name = step_config["name"]
        module = step_config["module"]
        expected_outputs: List[Path] = step_config["expected_outputs"]
        skip_flag = step_config["skip_flag"]
        
        logger.console("")
        logger.console(f"[{logger.timestamp()}] {step_name}")
        
        step_start = time.time()
        
        # Check if step should be skipped via flag
        if skip_flag == "skip_task4" and skip_task4:
            duration = time.time() - step_start
            status = "SKIPPED (--skip-task4)"
            logger.console(f"  → {status}")
            logger._log_to_file("")
            logger._log_to_file(f"STEP: {step_name}")
            logger._log_to_file(f"Status: {status}")
            results.append((step_name, status, duration, []))
            continue
        
        # Check if outputs already exist (skip unless --force)
        if not force:
            all_exist, missing = check_outputs_exist(expected_outputs)
            if all_exist:
                duration = time.time() - step_start
                status = "SKIPPED (outputs exist)"
                logger.console(f"  → {status}")
                logger._log_to_file("")
                logger._log_to_file(f"STEP: {step_name}")
                logger._log_to_file(f"Status: {status}")
                logger._log_to_file(f"Existing outputs: {len(expected_outputs)} files")
                results.append((step_name, status, duration, []))
                continue
        
        # Run the module
        logger.console(f"  → Running...")
        success, stdout, stderr = run_module(step_name, module, logger)
        
        # Log full output (NOT truncated)
        logger.log_step_output(stdout, stderr)
        
        duration = time.time() - step_start
        end_time = logger.timestamp()
        
        if success:
            # Verify outputs were created
            all_exist, missing = check_outputs_exist(expected_outputs)
            
            if all_exist:
                status = "SUCCESS"
                logger.console(f"  → {status} ({duration:.1f}s) - {len(expected_outputs)} outputs verified")
            else:
                status = "SUCCESS (with warnings)"
                logger.console(f"  → {status} ({duration:.1f}s) - {len(missing)} outputs missing")
                for m in missing:
                    logger.console(f"      Missing: {m.name}")
                all_missing_outputs.extend(missing)
            
            logger.log_step_end(status, duration, end_time)
            if missing:
                logger.log_missing_outputs(missing)
            
            results.append((step_name, status, duration, missing))
        else:
            status = "FAILED"
            logger.console(f"  → {status} ({duration:.1f}s)")
            # Print first few lines of error to console
            if stderr:
                for line in stderr.strip().split('\n')[:5]:
                    logger.console(f"      {line}")
            logger.log_step_end(status, duration, end_time)
            
            results.append((step_name, status, duration, []))
            
            # Stop pipeline on failure
            logger.console("")
            logger.console(f"⚠ PIPELINE STOPPED: {step_name} failed")
            logger._log_to_file("")
            logger._log_to_file(f"PIPELINE STOPPED: {step_name} failed")
            break
    
    # Pipeline summary
    pipeline_duration = time.time() - pipeline_start
    
    # Count results
    success_count = sum(1 for _, s, _, _ in results if "SUCCESS" in s)
    skipped_count = sum(1 for _, s, _, _ in results if "SKIPPED" in s)
    failed_count = sum(1 for _, s, _, _ in results if s == "FAILED")
    
    # Console summary
    logger.console("")
    logger.console("=" * 70)
    logger.console("PIPELINE SUMMARY")
    logger.console("=" * 70)
    logger.console(f"Finished: {logger.timestamp()}")
    logger.console(f"Duration: {pipeline_duration:.1f}s ({pipeline_duration/60:.1f} min)")
    logger.console("")
    logger.console(f"{'Step':<45} {'Status':<22} {'Time'}")
    logger.console("-" * 70)
    
    for step_name, status, duration, _ in results:
        short_name = step_name[:43] if len(step_name) > 43 else step_name
        short_status = status[:20] if len(status) > 20 else status
        logger.console(f"{short_name:<45} {short_status:<22} {duration:.1f}s")
    
    logger.console("-" * 70)
    logger.console(f"Total: {success_count} succeeded, {skipped_count} skipped, {failed_count} failed")
    
    # Log summary
    logger._log_to_file("")
    logger._log_to_file("=" * 80)
    logger._log_to_file("PIPELINE SUMMARY")
    logger._log_to_file("=" * 80)
    logger._log_to_file(f"Finished: {logger.timestamp()}")
    logger._log_to_file(f"Total duration: {pipeline_duration:.2f} seconds")
    logger._log_to_file("")
    logger._log_to_file(f"{'Step':<50} {'Status':<25} {'Duration'}")
    logger._log_to_file("-" * 80)
    
    for step_name, status, duration, _ in results:
        logger._log_to_file(f"{step_name:<50} {status:<25} {duration:.2f}s")
    
    logger._log_to_file("-" * 80)
    logger._log_to_file(f"Succeeded: {success_count}")
    logger._log_to_file(f"Skipped: {skipped_count}")
    logger._log_to_file(f"Failed: {failed_count}")
    
    # Log any missing outputs
    if all_missing_outputs:
        logger._log_to_file("")
        logger._log_to_file("=" * 80)
        logger._log_to_file("MISSING OUTPUT FILES")
        logger._log_to_file("=" * 80)
        for path in all_missing_outputs:
            logger._log_to_file(f"  - {path}")
    
    # Final status
    logger.console("")
    if failed_count == 0:
        logger.console("=" * 70)
        logger.console("✓ PIPELINE COMPLETED SUCCESSFULLY")
        logger.console("=" * 70)
        logger._log_to_file("")
        logger._log_to_file("FINAL STATUS: SUCCESS")
        exit_code = 0
    else:
        logger.console("=" * 70)
        logger.console("✗ PIPELINE FAILED")
        logger.console("=" * 70)
        logger._log_to_file("")
        logger._log_to_file("FINAL STATUS: FAILED")
        exit_code = 1
    
    # Save log
    log_path = logger.save()
    logger.console("")
    logger.console(f"Log saved: {log_path}")
    
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
