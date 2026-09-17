"""Batch-size sensitivity and fusion-operator comparison on the breast-cancer task (eps = 0.1)."""
import os, sys, json, time, pickle, dataclasses
import numpy as np
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(SCR, "patas-run")
sys.path.insert(0, os.path.join(REPO, "tests")); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "patas_module"))
import multiprocessing
from cancer_seed_sweep import omega_stats
OUT = os.path.join(SCR, "results_cancer_batch_fusion.json")

def main():
    from test_cancer import make_cancer_cfg, run_scenario
    from main import ptas_cache_dir
    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["kind"], r["seed"], r["value"], r["x_trust"], r["y_trust"]) for r in results}
    port = 5300
    for seed in (1, 2, 3):
        os.environ["PATAS_SEED"] = str(seed)
        os.environ["PATAS_CORRUPTION_SEED"] = str(20260816 + seed)
        # (a) batch-size sweep
        for bs in (8, 16, 32, 64, 128, 256):
            wd = os.path.join(SCR, "runs", f"cancer_bs{bs}_seed{seed}"); os.makedirs(wd, exist_ok=True); os.chdir(wd)
            for x, y in (("trust", "trust"), ("vacuous", "vacuous"), ("trust", "vacuous")):
                if ("batch", seed, bs, x, y) in done: continue
                port += 1
                cfg = make_cancer_cfg(x, y, epsilon_low=0.1, epochs=15, port=port)
                cfg = dataclasses.replace(cfg, batch_size=bs)
                t0 = time.time(); r = run_scenario(cfg)
                r.update({"kind": "batch", "seed": seed, "value": bs, "secs": time.time() - t0,
                          "omega": omega_stats(ptas_cache_dir("cancer", "16", x, y, 0.1))})
                results.append(r); json.dump(results, open(OUT, "w"), indent=1)
                print(f"[bs {bs} seed {seed}] {x}/{y}: TM={r['trust_mass']:.4f} test={r['test_acc']:.4f}", flush=True)
        # (b) fusion-operator sweep (revision operator)
        for fuse in ("average", "cumulative", "weighted", "constraint"):
            wd = os.path.join(SCR, "runs", f"cancer_fuse_{fuse}_seed{seed}"); os.makedirs(wd, exist_ok=True); os.chdir(wd)
            for x, y in (("trust", "trust"), ("vacuous", "vacuous"), ("trust", "vacuous")):
                if ("fuse", seed, fuse, x, y) in done: continue
                port += 1
                cfg = make_cancer_cfg(x, y, epsilon_low=0.1, epochs=15, port=port)
                cfg = dataclasses.replace(cfg, fuse_method=fuse)
                t0 = time.time(); r = run_scenario(cfg)
                r.update({"kind": "fuse", "seed": seed, "value": fuse, "secs": time.time() - t0,
                          "omega": omega_stats(ptas_cache_dir("cancer", "16", x, y, 0.1, fuse_method=fuse))})
                results.append(r); json.dump(results, open(OUT, "w"), indent=1)
                print(f"[fuse {fuse} seed {seed}] {x}/{y}: TM={r['trust_mass']:.4f}", flush=True)
    print("DONE")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
