import time
import tracemalloc
from functools import wraps

import numpy as np
import pandas as pd
from tqdm import tqdm


def _fast_copy(obj):
    """Fast shallow copy optimized for common data science objects."""
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.copy(deep=False)
    if isinstance(obj, np.ndarray):
        return obj.copy()
    return obj


def benchmark(
    repeats: int = 100,
    warmup: int = 0,
    desc: str | None = None,
    show_progress: bool = True,
    copy_per_iter: bool = False,
    measure_memory: bool = True,
):
    """Decorator for fast repeatable timing of a function with optional memory profiling."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            label = desc or func.__name__

            # Warmup runs
            if warmup > 0:
                for _ in range(warmup):
                    current_args = tuple(_fast_copy(a) for a in args) if copy_per_iter else args
                    func(*current_args, **kwargs)

            times = np.empty(repeats, dtype=np.float64)
            result = None

            iterator = tqdm(range(repeats), desc=f"{label:>16} bench", unit="run") if show_progress else range(repeats)

            # --- TIMING PHASE (no tracemalloc) ---
            for i in iterator:
                current_args = tuple(_fast_copy(a) for a in args) if copy_per_iter else args
                start = time.perf_counter()
                result = func(*current_args, **kwargs)
                times[i] = time.perf_counter() - start

            stats = {
                "runs": float(repeats),
                "median_time_s": float(np.median(times)),
                "std_time_s": float(np.std(times, ddof=0)),
                "min_time_s": float(np.min(times)),
            }

            # --- MEMORY PHASE (single call with tracemalloc) ---
            if measure_memory:
                current_args = tuple(_fast_copy(a) for a in args) if copy_per_iter else args
                tracemalloc.start()
                mem_before = tracemalloc.get_traced_memory()[0]
                func(*current_args, **kwargs)
                mem_current, mem_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                stats["peak_memory_mb"] = float(mem_peak / 1024 / 1024)
                stats["delta_memory_mb"] = float((mem_current - mem_before) / 1024 / 1024)

            return result, stats

        return wrapper

    return decorator


def compare_outputs(a, b, _depth=0):
    """Compare two outputs with recursive handling for nested structures."""
    # Prevent infinite recursion
    if _depth > 50:
        return False, "Maximum recursion depth exceeded"

    # Handle tuples and lists recursively
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        if len(a) != len(b):
            return False, f"Length mismatch: {len(a)} vs {len(b)}"

        for i, (elem_a, elem_b) in enumerate(zip(a, b, strict=True)):
            match, msg = compare_outputs(elem_a, elem_b, _depth + 1)
            if not match:
                return False, f"Element [{i}]: {msg}"

        return True, f"{type(a).__name__} elements identical"

    # Dict comparison
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            missing = set(b.keys()) - set(a.keys())
            extra = set(a.keys()) - set(b.keys())
            return False, f"Key mismatch: missing={missing}, extra={extra}"
        for key in sorted(a.keys(), key=str):
            match, msg = compare_outputs(a[key], b[key], _depth + 1)
            if not match:
                return False, f"Key '{key}': {msg}"
        return True, f"Dicts identical ({len(a)} keys)"

    # DataFrame comparison with attrs preservation
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        try:
            pd.testing.assert_frame_equal(
                a,
                b,
                check_exact=False,
                rtol=1e-9,
                atol=1e-12,
                check_flags=False,
            )
            # Compare attrs separately (not checked by assert_frame_equal)
            if a.attrs != b.attrs:
                return False, f"DataFrame attrs differ: {a.attrs} vs {b.attrs}"
            return True, "DataFrames identical"
        except AssertionError as e:
            return False, str(e).splitlines()[0][:100]

    # Series comparison — split scalar vs DataFrame entries (e.g. backtesting.py stats)
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        if a.index.tolist() != b.index.tolist():
            return False, f"Series index mismatch: {a.index.tolist()} vs {b.index.tolist()}"
        df_keys = [k for k in a.index if isinstance(a[k], pd.DataFrame)]
        scalar_keys = [k for k in a.index if k not in df_keys]
        if scalar_keys:
            try:
                pd.testing.assert_series_equal(a[scalar_keys], b[scalar_keys], check_exact=False, rtol=1e-9, atol=1e-12)
            except AssertionError as e:
                return False, str(e).splitlines()[0][:100]
        for k in df_keys:
            match, msg = compare_outputs(a[k], b[k], _depth + 1)
            if not match:
                return False, f"Series['{k}']: {msg}"
        return True, "Series identical"

    # NumPy arrays
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape:
            return False, f"Shape mismatch: {a.shape} vs {b.shape}"
        if np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True):
            return True, "Arrays identical"
        max_diff = np.max(np.abs(a - b))
        return False, f"Arrays differ (max |Δ|: {max_diff:.3e})"

    # Numeric types
    if isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        if np.isclose(a, b, rtol=1e-9, atol=1e-12):
            return True, "Numeric values equal"
        return False, f"Values differ: {a} vs {b}"

    try:
        return (True, "Outputs equal") if a == b else (False, "Outputs differ")
    except (ValueError, TypeError):
        return False, f"Cannot compare {type(a).__name__} vs {type(b).__name__}"


def _print_comparison(labels, stats_a, stats_b, show_memory=True):
    la, lb = labels
    med_a = stats_a["median_time_s"] * 1000
    med_b = stats_b["median_time_s"] * 1000
    ratio = med_a / med_b if med_b > 0 else float("inf")

    print(f"\n{'─' * 62}")
    print(f"{'':14}{la:>14} {lb:>14} {'ratio':>10}")
    print(f"{'─' * 62}")
    print(f"{'Median (ms)':14}{med_a:>14.3f} {med_b:>14.3f} {ratio:>9.2f}×")
    print(f"{'Min (ms)':14}{stats_a['min_time_s'] * 1000:>14.3f} {stats_b['min_time_s'] * 1000:>14.3f}")
    print(f"{'Std (ms)':14}{stats_a['std_time_s'] * 1000:>14.3f} {stats_b['std_time_s'] * 1000:>14.3f}")

    if show_memory and "peak_memory_mb" in stats_a and "peak_memory_mb" in stats_b:
        print(f"{'─' * 62}")
        peak_a = stats_a["peak_memory_mb"]
        peak_b = stats_b["peak_memory_mb"]
        delta_a = stats_a["delta_memory_mb"]
        delta_b = stats_b["delta_memory_mb"]
        mem_ratio = peak_a / peak_b if peak_b > 0 else float("inf")

        print(f"{'Peak Mem (MB)':14}{peak_a:>14.2f} {peak_b:>14.2f} {mem_ratio:>9.2f}×")
        print(f"{'Delta Mem (MB)':14}{delta_a:>14.2f} {delta_b:>14.2f}")

    print(f"{'─' * 62}\n")


def compare_functions(
    fn_a,
    fn_b,
    *args,
    repeats: int = 50,
    warmup: int = 3,
    labels: tuple[str, str] = ("Old", "New"),
    copy_per_iter: bool = True,
    measure_memory: bool = True,
    **kwargs,
):
    """Benchmark and compare two functions for performance, memory, and correctness."""
    results = {}

    for label, fn in zip(labels, (fn_a, fn_b), strict=True):
        bench_fn = benchmark(
            repeats=repeats,
            warmup=warmup,
            desc=label,
            copy_per_iter=copy_per_iter,
            measure_memory=measure_memory,
        )(fn)
        result, stats = bench_fn(*args, **kwargs)
        results[label] = {"result": result, "stats": stats}

    # Output comparison
    result_a = results[labels[0]]["result"]
    result_b = results[labels[1]]["result"]
    match, msg = compare_outputs(result_a, result_b)

    # Print performance comparison
    _print_comparison(labels, results[labels[0]]["stats"], results[labels[1]]["stats"], measure_memory)

    # Print correctness check with enhanced diagnostics
    symbol = "✓" if match else "✗"
    print(f"Output match: {symbol} {msg}")

    if not match:
        print("\n⚠️  MISMATCH DETAILS:")
        print(f"    Type A: {type(result_a)}")
        print(f"    Type B: {type(result_b)}")
        if isinstance(result_a, tuple) and isinstance(result_b, tuple):
            print(f"    Tuple lengths: {len(result_a)} vs {len(result_b)}")

    # Validation summary
    if match and measure_memory:
        old_time = results[labels[0]]["stats"]["median_time_s"]
        new_time = results[labels[1]]["stats"]["median_time_s"]
        old_mem = results[labels[0]]["stats"]["peak_memory_mb"]
        new_mem = results[labels[1]]["stats"]["peak_memory_mb"]
        print("\n✓ Optimization validated:")
        print(f"  • Speed:  {old_time / new_time:.1f}× faster")
        print(f"  • Memory: {old_mem / new_mem:.2f}× peak ratio")
        print("  • Output: Identical")

    # Add match info to results
    results["match"] = {"is_equal": match, "message": msg}

    return results
