from pathlib import Path
import re
import statistics
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_output/openfrontcluttered_8traj_3seed"
OUT.mkdir(parents=True, exist_ok=True)

logs = {
    "Baseline": [
        ROOT / "logs/openfrontcluttered_baseline_small_20260622.log",
        ROOT / "logs/openfrontcluttered_baseline_seed1_20260622.log",
        ROOT / "logs/openfrontcluttered_baseline_seed2_20260622.log",
    ],
    "PACT": [
        ROOT / "logs/openfrontcluttered_pact_small_20260622.log",
        ROOT / "logs/openfrontcluttered_pact_seed1_20260622.log",
        ROOT / "logs/openfrontcluttered_pact_seed2_20260622.log",
    ],
}

def parse(path):
    text = path.read_text(errors="ignore")
    train = {
        int(e): float(v)
        for e, v in re.findall(r"\[epoch (\d+)\] train_loss=([0-9.]+)", text)
    }
    val = {
        int(e): float(v)
        for e, v in re.findall(r"\[epoch (\d+)\] val_loss=([0-9.]+)", text)
    }
    done = re.findall(
        r"\[done\] best val_loss=([0-9.]+) at epoch (\d+)", text
    )
    return train, val, done[-1] if done else None

parsed = {
    method: [parse(p) for p in paths]
    for method, paths in logs.items()
}

# 每个 seed 的 baseline vs PACT validation curve
for seed in range(3):
    plt.figure(figsize=(8, 5))
    for method in ("Baseline", "PACT"):
        val = parsed[method][seed][1]
        xs = sorted(val)
        plt.plot(xs, [val[x] for x in xs], label=method)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title(f"OpenFrontCluttered validation loss — seed {seed}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / f"val_loss_seed_{seed}.png", dpi=200)
    plt.close()

# 三个 seed 的平均 validation curve
plt.figure(figsize=(8, 5))
for method in ("Baseline", "PACT"):
    curves = [x[1] for x in parsed[method]]
    common = sorted(set.intersection(*(set(c) for c in curves)))
    mean = [statistics.mean(c[e] for c in curves) for e in common]
    plt.plot(common, mean, label=method)
plt.xlabel("Epoch")
plt.ylabel("Mean validation loss")
plt.title("OpenFrontCluttered mean validation loss — 3 seeds")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "val_loss_mean_3seeds.png", dpi=200)
plt.close()

# 最佳 validation loss 对比
best = {}
for method in ("Baseline", "PACT"):
    vals = [float(x[2][0]) for x in parsed[method]]
    best[method] = vals

means = [statistics.mean(best[m]) for m in ("Baseline", "PACT")]
stds = [statistics.stdev(best[m]) for m in ("Baseline", "PACT")]

plt.figure(figsize=(6, 5))
plt.bar(["Baseline", "PACT"], means, yerr=stds, capsize=6)
plt.ylabel("Best validation loss")
plt.title("Best validation loss — mean ± std, 3 seeds")
plt.tight_layout()
plt.savefig(OUT / "best_val_loss_comparison.png", dpi=200)
plt.close()

summary = OUT / "summary.txt"
summary.write_text(
    f"Baseline: {means[0]:.5f} ± {stds[0]:.5f}\n"
    f"PACT: {means[1]:.5f} ± {stds[1]:.5f}\n"
    f"Mean reduction: {(means[0] - means[1]) / means[0] * 100:.2f}%\n"
)

print("Wrote:")
for p in sorted(OUT.iterdir()):
    print(p)
