"""
End-to-end pipeline runner.

Executes all pipeline steps in order:
1. load_articles
2. coref_contexts
3. extract_descriptions
4. embed_cluster
5. freq_tables
6. stats_tests

Saves logs to outputs/reports/run_log.txt
"""

import sys
import time
from datetime import datetime
from io import StringIO
from pathlib import Path

from . import config


class TeeOutput:
    """Capture stdout to both console and string buffer."""
    
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = StringIO()
    
    def write(self, text):
        self.original_stdout.write(text)
        self.buffer.write(text)
    
    def flush(self):
        self.original_stdout.flush()
    
    def get_log(self):
        return self.buffer.getvalue()


def run_step(step_name: str, module_name: str) -> bool:
    """
    Run a single pipeline step.
    
    Args:
        step_name: Human-readable step name
        module_name: Module to import and run main() from
        
    Returns:
        True if successful, False otherwise
    """
    print("\n" + "=" * 70)
    print(f"STEP: {step_name}")
    print(f"Module: {module_name}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    
    try:
        # Dynamically import and run the module's main function
        module = __import__(module_name, fromlist=['main'])
        module.main()
        
        elapsed = time.time() - start_time
        print(f"\n✓ {step_name} completed in {elapsed:.1f} seconds")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ {step_name} FAILED after {elapsed:.1f} seconds")
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Run the complete NLP pipeline end-to-end.
    Run with: python -m src.run_all
    """
    # Capture output to log
    original_stdout = sys.stdout
    tee = TeeOutput(original_stdout)
    sys.stdout = tee
    
    # Start timing
    pipeline_start = time.time()
    start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("=" * 70)
    print("NLP HOMEWORK 5 - FULL PIPELINE EXECUTION")
    print("=" * 70)
    print(f"Started at: {start_timestamp}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {Path.cwd()}")
    print("=" * 70)
    
    # Ensure output directories exist
    config.ensure_output_dirs()
    
    # Define pipeline steps
    steps = [
        ("1. Load Articles", "src.load_articles"),
        ("2. Coreference Resolution & Context Extraction", "src.coref_contexts"),
        ("3. Description Extraction", "src.extract_descriptions"),
        ("4. Embedding & Clustering", "src.embed_cluster"),
        ("5. Frequency Tables", "src.freq_tables"),
        ("6. Statistical Tests", "src.stats_tests"),
    ]
    
    # Run each step
    results = []
    for step_name, module_name in steps:
        success = run_step(step_name, module_name)
        results.append((step_name, success))
        
        if not success:
            print(f"\n⚠ Pipeline stopped due to failure in: {step_name}")
            break
    
    # Pipeline summary
    pipeline_elapsed = time.time() - pipeline_start
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"Finished at: {end_timestamp}")
    print(f"Total time: {pipeline_elapsed:.1f} seconds ({pipeline_elapsed/60:.1f} minutes)")
    print("\nStep Results:")
    
    all_success = True
    for step_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {step_name}")
        if not success:
            all_success = False
    
    if all_success:
        print("\n" + "=" * 70)
        print("✓ ALL STEPS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nOutputs created:")
        print("  Processed data:  outputs/processed/")
        print("  Figures:         outputs/figures/")
        print("  Reports:         outputs/reports/")
    else:
        print("\n" + "=" * 70)
        print("⚠ PIPELINE COMPLETED WITH ERRORS")
        print("=" * 70)
    
    # Restore stdout and save log
    sys.stdout = original_stdout
    
    log_content = tee.get_log()
    log_path = config.REPORTS_DIR / "run_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    print(f"\nLog saved to: {log_path}")
    
    # Return exit code
    return 0 if all_success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

