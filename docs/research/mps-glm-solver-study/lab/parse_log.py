"""Rebuild a matrix-results JSON from a run_matrix .log (the runs were stopped
before the final JSON write). Line format:
  [regime    solver    repN] VERDICT      err=... nd=... outer=NN  W.Ws
"""
import re, json, sys

LINE = re.compile(
    r"\[(\w+)\s+(\w+)\s+rep(\d+)\]\s+(\w+)\s+err=([\d.eE+-]+|nan)\s+"
    r"nd=([\d.eE+-]+|nan)\s+outer=(\d+)\s+([\d.]+)s")

def parse(path):
    out = []
    for line in open(path):
        m = LINE.search(line)
        if not m:
            continue
        reg, sol, rep, verdict, err, nd, outer, wall = m.groups()
        f = lambda x: float("nan") if x == "nan" else float(x)
        out.append({"regime": reg, "solver": sol, "rep": int(rep),
                    "verdict": verdict, "coef_max_rel_err": f(err),
                    "rel_newton_dec": f(nd), "outer_iters": int(outer),
                    "wall_s": float(wall), "raised": verdict == "fail_loud" and err == "nan"})
    return out

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    res = parse(src)
    json.dump({"results": res}, open(dst, "w"), indent=2)
    print(f"{src}: parsed {len(res)} runs -> {dst}")
