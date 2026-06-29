"""Aggregate matrix_mps.json / matrix_cuda.json / fp64_ref.json / intermittency
jsonl into the markdown evidence tables for the research report."""
import json, sys, os
from collections import Counter, defaultdict

def load(p):
    return json.load(open(p)) if os.path.exists(p) else None

def fmt_err(e):
    if e is None: return "—"
    try:
        return f"{e:.2e}" if e == e else "crash"  # nan -> crash
    except Exception:
        return str(e)

def per_combo(results):
    g = defaultdict(list)
    for r in results:
        g[(r["regime"], r["solver"])].append(r)
    return g

REG_ORDER = ["yearly","quarterly","monthly","biweekly"]
SOL_ORDER = ["cholesky","cg","lm","hybrid","cpu_qr"]

def matrix_table(path, title):
    data = load(path)
    if not data:
        return f"\n### {title}\n\n_(not available)_\n"
    g = per_combo(data["results"])
    out = [f"\n### {title}\n",
           "| solver | regime | n | converged | fail_loud | silent_wrong | coef_err (min..max) | med wall (s) | outer | cg it/step |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for sol in SOL_ORDER:
        for reg in REG_ORDER:
            runs = g.get((reg, sol))
            if not runs: continue
            v = Counter(r["verdict"] for r in runs)
            errs = [r["coef_max_rel_err"] for r in runs
                    if isinstance(r["coef_max_rel_err"],(int,float)) and r["coef_max_rel_err"]==r["coef_max_rel_err"]]
            erng = f"{min(errs):.2e}..{max(errs):.2e}" if errs else "crash"
            walls = sorted(r["wall_s"] for r in runs)
            med = walls[len(walls)//2]
            outer = runs[0]["outer_iters"]
            cg = [r.get("cg_iters_mean") for r in runs if r.get("cg_iters_mean")]
            cgs = f"{sum(cg)/len(cg):.0f}" if cg else "—"
            out.append(f"| {sol} | {reg} | {len(runs)} | {v.get('converged',0)} | "
                       f"{v.get('fail_loud',0)} | {v.get('silent_wrong',0)} | {erng} | "
                       f"{med:.2f} | {outer} | {cgs} |")
    return "\n".join(out) + "\n"

def intermittency_table(path, title):
    if not os.path.exists(path):
        return f"\n### {title}\n\n_(not available)_\n"
    rows=[json.loads(l) for l in open(path) if l.strip()]
    g=defaultdict(list)
    for r in rows: g[(r["regime"],r["solver"])].append(r)
    out=[f"\n### {title}\n",
         "| solver | regime | launches | converged | fail_loud(crash/refuse) | silent_wrong |",
         "|---|---|---|---|---|---|"]
    for (reg,sol),rs in sorted(g.items()):
        c=Counter(r["verdict"] for r in rs)
        out.append(f"| {sol} | {reg} | {len(rs)} | {c.get('converged',0)} | "
                   f"{c.get('fail_loud',0)} | {c.get('silent_wrong',0)} |")
    return "\n".join(out)+"\n"

def fp64_table(path):
    data=load(path)
    if not data: return "\n### CUDA gpu_fp64 vs CPU fp64\n\n_(not available)_\n"
    out=["\n### CUDA gpu_fp64 vs CPU fp64 (exact reference)\n",
         "| regime | rows x cols | gpu_fp64 max rel err vs CPU fp64 |","|---|---|---|"]
    for reg in REG_ORDER:
        if reg in data:
            d=data[reg]
            out.append(f"| {reg} | {d['n_rows']}x{d['n_cols']} | {d['gpu_fp64_vs_cpu_fp64_max_rel_err']:.2e} |")
    return "\n".join(out)+"\n"

if __name__=="__main__":
    base=sys.argv[1] if len(sys.argv)>1 else "."
    print(matrix_table(os.path.join(base,"matrix_mps.json"),
                       "MPS float32 — within-process matrix (6 reps each)"))
    print(intermittency_table(os.path.join(base,"intermittency_mps.jsonl"),
                       "MPS float32 — cross-process intermittency (independent launches)"))
    print(matrix_table(os.path.join(base,"matrix_cuda.json"),
                       "CUDA float32 — within-process matrix (6 reps each)"))
    print(intermittency_table(os.path.join(base,"intermittency_cuda.jsonl"),
                       "CUDA float32 — cross-process intermittency"))
    print(fp64_table(os.path.join(base,"fp64_ref.json")))
