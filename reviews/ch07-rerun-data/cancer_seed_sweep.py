"""Multi-seed re-run of the breast-cancer grid (Table 7.1) with the released implementation.
Each (seed, eps) pair runs in a fresh working directory so that PaTAS always consumes the
genuine training gradient stream (no continued-training replay)."""
import os, sys, json, time, pickle, glob
import numpy as np
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(SCR, "patas-run")
sys.path.insert(0, os.path.join(REPO, "tests")); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "patas_module"))
import multiprocessing

SEEDS = [1, 2, 3, 4, 5]
EPS = [1, 0.1, 0.01, 0.001]
GRID = [(x, y) for x in ("trust", "vacuous", "distrust") for y in ("trust", "vacuous", "distrust")]
EXTRA = [("0.25,0.25,0.5", "0.25,0.25,0.5", "noise", "noise"),
         ("0.25,0.0,0.75", "0.25,0.0,0.75", "noise_mild", "noise_mild")]
OUT = os.path.join(SCR, "results_cancer_seeds.json")

def omega_stats(ptas_dir):
    p = os.path.join(ptas_dir, "omega_arrays.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as fh:
        arrs = pickle.load(fh)
    stats = []
    for a in arrs:
        a = np.asarray(a, dtype=np.float64)
        b, d, u = a[..., 0], a[..., 1], a[..., 2]
        stats.append({"shape": list(a.shape), "b_mean": float(b.mean()), "d_mean": float(d.mean()),
                      "u_mean": float(u.mean()), "u_median": float(np.median(u)),
                      "u_max": float(u.max()), "frac_u_lt_1e-3": float((u < 1e-3).mean()),
                      "p_mean": float((b + 0.5 * u).mean())})
    return stats

def main():
    from test_cancer import make_cancer_cfg, run_scenario
    from main import ptas_cache_dir
    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["seed"], r["eps"], r["x_trust"], r["y_trust"]) for r in results}
    port = 5100
    for seed in SEEDS:
        os.environ["PATAS_SEED"] = str(seed)
        os.environ["PATAS_CORRUPTION_SEED"] = str(20260816 + seed)
        for eps in EPS:
            wd = os.path.join(SCR, "runs", f"cancer_seed{seed}_eps{eps}")
            os.makedirs(wd, exist_ok=True); os.chdir(wd)
            scen = [(x, y, None, None) for x, y in GRID] + EXTRA
            for (x, y, xd, yd) in scen:
                if (seed, eps, x, y) in done:
                    continue
                port += 1
                cfg = make_cancer_cfg(x, y, epsilon_low=eps, epochs=15, port=port, x_dataset=xd, y_dataset=yd)
                t0 = time.time()
                r = run_scenario(cfg)
                r.update({"seed": seed, "eps": eps, "x_dataset": xd, "y_dataset": yd, "secs": time.time() - t0})
                r["omega"] = omega_stats(ptas_cache_dir("cancer", "16", x, y, eps))
                results.append(r)
                json.dump(results, open(OUT, "w"), indent=1)
                print(f"[seed {seed} eps {eps}] {x}/{y}: TM={r['trust_mass']:.4f} train={r['train_acc']:.4f} test={r['test_acc']:.4f} ({r['secs']:.0f}s)", flush=True)
                time.sleep(1)
    print("DONE")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
