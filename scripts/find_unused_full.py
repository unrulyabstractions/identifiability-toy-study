#!/usr/bin/env python3
"""Find unused functions by tracing with full visualization."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
import ast
import shutil

CALLED_FUNCTIONS = set()
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"


def trace_calls(frame, event, arg):
    if event != 'call':
        return trace_calls
    code = frame.f_code
    filename = code.co_filename
    funcname = code.co_name
    if str(SRC_DIR) in filename and '/spd/' not in filename:
        rel_path = os.path.relpath(filename, PROJECT_ROOT)
        CALLED_FUNCTIONS.add((rel_path, funcname))
    return trace_calls


def get_all_functions():
    all_functions = {}
    for py_file in SRC_DIR.rglob("*.py"):
        if "/spd/" in str(py_file):
            continue
        rel_path = str(py_file.relative_to(PROJECT_ROOT))
        try:
            with open(py_file, 'r') as f:
                tree = ast.parse(f.read())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    all_functions[(rel_path, node.name)] = node.lineno
        except:
            pass
    return all_functions


def main():
    all_functions = get_all_functions()
    print(f"Total functions defined: {len(all_functions)}")

    # Run with full viz
    print("Running with --viz 3...")
    sys.settrace(trace_calls)
    sys.argv = ['main.py', '--test', '0', '--rename', 'trace_full', '--viz', '3']
    try:
        import main as main_module
        main_module.main()
    except SystemExit:
        pass
    finally:
        sys.settrace(None)

    print(f"\nFunctions called: {len(CALLED_FUNCTIONS)}")

    # Find uncalled
    uncalled = [(fp, fn, ln) for (fp, fn), ln in all_functions.items()
                if (fp, fn) not in CALLED_FUNCTIONS]
    print(f"Functions NOT called: {len(uncalled)}")

    # Group by directory
    by_dir = {}
    for fp, fn, ln in uncalled:
        parts = fp.split('/')
        if len(parts) >= 2:
            dir_name = '/'.join(parts[:2])
        else:
            dir_name = parts[0]
        by_dir.setdefault(dir_name, []).append((fp, fn, ln))

    # Check for entire unused directories
    print("\n" + "="*70)
    print("LIKELY DELETABLE MODULES (all functions uncalled):")
    print("="*70)

    for dir_name in sorted(by_dir.keys()):
        funcs = by_dir[dir_name]
        # Count total functions in this dir
        total_in_dir = sum(1 for (fp, _), _ in all_functions.items()
                          if fp.startswith(dir_name + '/') or fp == dir_name)
        if total_in_dir == len(funcs) and total_in_dir >= 3:
            print(f"\n{dir_name}/ - {len(funcs)} functions, ALL uncalled")
            for fp, fn, ln in sorted(funcs)[:5]:
                print(f"    {fn}")
            if len(funcs) > 5:
                print(f"    ... and {len(funcs)-5} more")

    # Save detailed report
    with open(PROJECT_ROOT / "uncalled_full.txt", 'w') as f:
        for fp, fn, ln in sorted(uncalled):
            f.write(f"{fp}:{ln}: {fn}\n")
    print(f"\nDetailed report: uncalled_full.txt")

    # Cleanup
    trace_dir = PROJECT_ROOT / "runs" / "trace_full"
    if trace_dir.exists():
        shutil.rmtree(trace_dir)


if __name__ == "__main__":
    main()
