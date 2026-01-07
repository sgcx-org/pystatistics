# bench_multiple_sizes.py
import numpy as np
import time
from pystatistics.regression.design import Design
from pystatistics.regression.backends.cpu import CPUQRBackend

def bench_pystatistics(n, p, trials=5):
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    
    design = Design.from_arrays(X=X, y=y)
    backend = CPUQRBackend()
    
    # Warmup
    _ = backend.solve(design)
    
    times = []
    for _ in range(trials):
        start = time.perf_counter()
        result = backend.solve(design)
        times.append(time.perf_counter() - start)
    
    return np.median(times)

print("PyStatistics CPU Benchmark")
print("="*60)
for n, p in [(1000, 10), (10000, 50), (100000, 100), (500000, 200), (1000000, 500)]:
    t = bench_pystatistics(n, p)
    print(f"n={n:>7}, p={p:>4}: {t*1000:>8.2f} ms")