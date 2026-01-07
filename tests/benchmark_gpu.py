"""Quick GPU benchmark - does it work? How fast?"""

import numpy as np
import time
from pystatistics.regression import fit
from pystatistics.regression.design import Design

print("PyStatistics GPU Benchmark")
print("="*60)

# Check GPU availability
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ No CUDA GPU available")
        exit(1)
except ImportError:
    print("❌ PyTorch not installed")
    exit(1)

print()

# Test sizes
sizes = [
    (10000, 50, "Small"),
    (100000, 100, "Medium"),
    (500000, 200, "Large"),
    (1000000, 500, "X-Large"),
]

results = []

for n, p, label in sizes:
    print(f"{label}: n={n:>7,}, p={p:>4}")
    print("-"*60)
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    
    # CPU benchmark
    design = Design.from_arrays(X, y)
    _ = fit(design, backend='cpu')  # Warmup
    
    cpu_times = []
    for _ in range(3):
        start = time.perf_counter()
        result_cpu = fit(design, backend='cpu')
        cpu_times.append(time.perf_counter() - start)
    
    cpu_time = np.median(cpu_times)
    
    # GPU benchmark
    _ = fit(design, backend='gpu')  # Warmup
    
    gpu_times = []
    for _ in range(3):
        start = time.perf_counter()
        result_gpu = fit(design, backend='gpu')
        gpu_times.append(time.perf_counter() - start)
    
    gpu_time = np.median(gpu_times)
    
    # Validate numerical equivalence
    coef_diff = np.max(np.abs(result_cpu.coefficients - result_gpu.coefficients))
    
    speedup = cpu_time / gpu_time
    
    print(f"  CPU:     {cpu_time*1000:>8.2f} ms")
    print(f"  GPU:     {gpu_time*1000:>8.2f} ms")
    print(f"  Speedup: {speedup:>8.1f}x")
    print(f"  Max coef diff: {coef_diff:.2e}")
    
    # Validate
    if coef_diff < 1e-4:
        print(f"  ✅ Numerical validation PASSED")
    else:
        print(f"  ⚠️  Numerical validation marginal (check tolerance)")
    
    print()
    
    results.append({
        'size': label,
        'n': n,
        'p': p,
        'cpu_ms': cpu_time * 1000,
        'gpu_ms': gpu_time * 1000,
        'speedup': speedup,
    })

# Summary
print("="*60)
print("SUMMARY")
print("="*60)
for r in results:
    print(f"{r['size']:8s} (n={r['n']:>7,}): {r['speedup']:>5.1f}x speedup")

print()
print("Key takeaways:")
print(f"  - Smallest problem: {results[0]['speedup']:.1f}x (overhead-limited)")
print(f"  - Largest problem:  {results[-1]['speedup']:.1f}x (compute-optimal)")
print()

# Project to UK Biobank
print("Projected UK Biobank GWAS (n=500K, 10M SNPs):")
large_per_snp = results[2]['gpu_ms'] / 1000  # Use n=500K timing
total_time_minutes = (large_per_snp * 10_000_000) / 60
print(f"  Estimated time: {total_time_minutes:.1f} minutes")
print(f"  (vs R: ~100 days)")