"""Multi-seed re-run of the MNIST architecture table (Table 7.3) and fusion-operator variants on 784-128-10.
Every run uses a fresh working directory so PaTAS consumes the genuine training gradient stream."""
import os, sys, json, time, dataclasses
SCR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.join(SCR, "patas-run")
sys.path.insert(0, os.path.join(REPO, "tests")); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "patas_module"))
import multiprocessing
from cancer_seed_sweep import omega_stats
OUT = os.path.join(SCR, "results_mnist_sweep.json")
CONFIGS = [((16,), "vacuous", "vacuous"), ((32,), "vacuous", "vacuous"), ((64,), "vacuous", "vacuous"),
           ((128,), "vacuous", "vacuous"), ((16, 16), "vacuous", "vacuous"), ((16,), "trust", "trust"), ((16, 16), "trust", "trust")]

def main():
    from test_mnist import make_mnist_cfg, run_scenario
    from main import ptas_cache_dir
    results = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["kind"], r["seed"], r["arch"], r["x_trust"], r["y_trust"], r.get("fuse", "average")) for r in results}
    port = 5500
    for seed in (1, 2, 3):
        os.environ["PATAS_SEED"] = str(seed); os.environ["PATAS_CORRUPTION_SEED"] = str(20260816 + seed)
        for hd, x, y in CONFIGS:
            arch = "-".join(map(str, hd))
            if ("seed", seed, arch, x, y, "average") in done: continue
            wd = os.path.join(SCR, "runs", f"mnist_{arch}_{x}_{y}_seed{seed}"); os.makedirs(wd, exist_ok=True); os.chdir(wd)
            port += 1
            cfg = make_mnist_cfg(x, y, epsilon_low=0.05, epochs=20, hidden_dims=hd, port=port)
            t0 = time.time(); r = run_scenario(cfg)
            r.update({"kind": "seed", "seed": seed, "arch": arch, "x_trust": x, "y_trust": y, "fuse": "average", "secs": time.time() - t0,
                      "omega": omega_stats(ptas_cache_dir("mnist", "_".join(map(str, hd)), x, y, 0.05))})
            results.append(r); json.dump(results, open(OUT, "w"), indent=1)
            print(f"[seed {seed}] {arch} {x}/{y}: TM={r['trust_mass']:.4f} train={r['train_acc']:.4f} test={r['test_acc']:.4f} ({r['secs']:.0f}s)", flush=True)
    # fusion-operator variants on 784-128-10, trust/trust and vacuous/vacuous, seed 1
    os.environ["PATAS_SEED"] = "1"; os.environ["PATAS_CORRUPTION_SEED"] = str(20260816 + 1)
    for fuse in ("average", "cumulative", "weighted", "constraint"):
        for x, y in (("trust", "trust"), ("vacuous", "vacuous")):
            if ("fuse", 1, "128", x, y, fuse) in done: continue
            wd = os.path.join(SCR, "runs", f"mnist_128_{x}_{y}_fuse_{fuse}_seed1"); os.makedirs(wd, exist_ok=True); os.chdir(wd)
            port += 1
            cfg = make_mnist_cfg(x, y, epsilon_low=0.05, epochs=20, hidden_dims=(128,), port=port)
            cfg = dataclasses.replace(cfg, fuse_method=fuse)
            t0 = time.time(); r = run_scenario(cfg)
            r.update({"kind": "fuse", "seed": 1, "arch": "128", "x_trust": x, "y_trust": y, "fuse": fuse, "secs": time.time() - t0,
                      "omega": omega_stats(ptas_cache_dir("mnist", "128", x, y, 0.05, fuse_method=fuse))})
            results.append(r); json.dump(results, open(OUT, "w"), indent=1)
            print(f"[fuse {fuse}] 128 {x}/{y}: TM={r['trust_mass']:.4f} ({r['secs']:.0f}s)", flush=True)
    print("DONE")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
