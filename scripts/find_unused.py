#!/usr/bin/env python3
"""Find unused functions by tracing execution.

Usage:
    python scripts/find_unused.py

Runs main.py with tracing, then reports functions that were never called.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import ast

# Track called functions: {(filename, funcname)}
CALLED_FUNCTIONS = set()
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def trace_calls(frame, event, arg):
    """Trace function to track all calls."""
    if event != 'call':
        return trace_calls

    code = frame.f_code
    filename = code.co_filename
    funcname = code.co_name

    # Only track functions in our src/ directory (excluding spd/)
    if str(SRC_DIR) in filename and '/spd/' not in filename:
        # Normalize path
        rel_path = os.path.relpath(filename, PROJECT_ROOT)
        CALLED_FUNCTIONS.add((rel_path, funcname))

    return trace_calls


def get_all_functions():
    """Get all function definitions in src/ (excluding spd/)."""
    all_functions = {}  # {(filepath, funcname): lineno}

    for py_file in SRC_DIR.rglob("*.py"):
        # Skip spd/
        if "/spd/" in str(py_file) or "\\spd\\" in str(py_file):
            continue

        rel_path = str(py_file.relative_to(PROJECT_ROOT))

        try:
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read(), filename=str(py_file))
        except SyntaxError:
            print(f"Warning: Could not parse {rel_path}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                funcname = node.name
                lineno = node.lineno
                all_functions[(rel_path, funcname)] = lineno

    return all_functions


def main():
    print("=" * 70)
    print("STEP 1: Finding all function definitions in src/ (excluding spd/)")
    print("=" * 70)

    all_functions = get_all_functions()
    print(f"Found {len(all_functions)} function definitions")

    print("\n" + "=" * 70)
    print("STEP 2: Running main.py --test 0 with tracing")
    print("=" * 70)

    # Set up tracing
    sys.settrace(trace_calls)

    # Import and run main
    try:
        # Simulate running: python main.py --test 0
        sys.argv = ['main.py', '--test', '0', '--rename', 'trace_run']

        # Import main module
        import main as main_module
        main_module.main()

    except SystemExit:
        pass  # main() calls sys.exit()
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.settrace(None)

    print("\n" + "=" * 70)
    print("STEP 3: Analyzing called vs defined functions")
    print("=" * 70)

    print(f"Functions called: {len(CALLED_FUNCTIONS)}")

    # Find uncalled functions
    uncalled = []
    for (filepath, funcname), lineno in sorted(all_functions.items()):
        if (filepath, funcname) not in CALLED_FUNCTIONS:
            uncalled.append((filepath, funcname, lineno))

    print(f"Functions NOT called: {len(uncalled)}")

    print("\n" + "=" * 70)
    print("UNCALLED FUNCTIONS (candidates for removal)")
    print("=" * 70)

    # Group by file
    by_file = {}
    for filepath, funcname, lineno in uncalled:
        by_file.setdefault(filepath, []).append((funcname, lineno))

    for filepath in sorted(by_file.keys()):
        print(f"\n{filepath}:")
        for funcname, lineno in sorted(by_file[filepath], key=lambda x: x[1]):
            # Skip special methods and test helpers
            if funcname.startswith('_') and not funcname.startswith('__'):
                marker = "  (private)"
            elif funcname.startswith('test_'):
                marker = "  (test)"
            else:
                marker = ""
            print(f"  line {lineno:4d}: {funcname}{marker}")

    # Save to file
    output_file = PROJECT_ROOT / "uncalled_functions.txt"
    with open(output_file, 'w') as f:
        f.write("UNCALLED FUNCTIONS\n")
        f.write("=" * 70 + "\n\n")
        for filepath in sorted(by_file.keys()):
            f.write(f"{filepath}:\n")
            for funcname, lineno in sorted(by_file[filepath], key=lambda x: x[1]):
                f.write(f"  line {lineno:4d}: {funcname}\n")
            f.write("\n")

    print(f"\n\nSaved to: {output_file}")

    # Cleanup test run
    import shutil
    trace_run = PROJECT_ROOT / "runs" / "trace_run"
    if trace_run.exists():
        shutil.rmtree(trace_run)
        print("Cleaned up trace_run directory")


if __name__ == "__main__":
    main()
