import sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from kg_filter import *

def load(path=None):
    if path:
        return [float(x) for x in open(path).read().replace(",", " ").split()]
    return [10, 15, 13, 19, 14, 18, 17, 11]

F = load(sys.argv[1] if len(sys.argv) > 1 else None)
n = len(F)
TOL = 0.15
res = {}
print("Ряд:", F, "n =", n)

# 1) Навчання за всією вибіркою
print("\n== 1. Навчання за всіма точками ==")
for name, idx in MODELS.items():
    a, N, b, ex = fit(F, idx)
    A, y = design(F, idx)
    s = skv(F, a, idx)
    nxt = predict_one(a, idx, F[-2], F[-1])
    res[name] = dict(a=a.tolist(), skv=s, windows=len(y), ncoef=len(idx), exact=bool(ex), next=nxt)
    print(f"{name:32s} рівнянь={len(y)} коеф.={len(idx)} СКВ={s:.4f} F(n+1)={nxt:.3f}")
    print("   a =", np.round(a, 4))

# 2) Контроль на прогнозі: навчання за n-2 точками, прогноз 2 останніх
print("\n== 2. Навчання за n-2 точками, прогноз двох останніх ==")
m = n - 2
for name, idx in MODELS.items():
    if m - 2 < len(idx):
        print(f"{name:32s} недостатньо даних"); continue
    a, *_ = fit(F[:m], idx)
    pr = forecast_recursive(F[:m], a, idx, 2)
    errs = [abs(p - f) for p, f in zip(pr, F[m:])]
    res[name]["holdout"] = dict(pred=pr, err=errs, skv_train=skv(F[:m], a, idx))
    print(f"{name:32s} прогноз={np.round(pr,3)} факт={F[m:]} похибки={np.round(errs,3)} СКВ(навч.)={skv(F[:m],a,idx):.3f}")

# 3) Горизонт прогнозування
print(f"\n== 3. Горизонт прогнозування (допуск {TOL:.0%}) ==")
for name, idx in MODELS.items():
    H, log = horizon(F, idx, TOL)
    res[name]["horizon"] = H
    res[name]["horizon_log"] = log
    print(name, "-> горизонт", H)
    for r in log:
        print("   k=%d навч.точок=%d прогноз=%s похибка=%s %s" % (r[0], r[1],
              "-" if r[2] is None else f"{r[2]:.3f}", "-" if r[3] is None else f"{r[3]:.1%}", r[4]))

json.dump(res, open("results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

# Графіки
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
t = np.arange(1, n + 1)
ax[0].plot(t, F, "o-", color="#c0392b", lw=2, label="Дійсні значення")
for name, idx in list(MODELS.items())[:3]:
    a, *_ = fit(F, idx)
    fit_vals = [predict_one(a, idx, F[i], F[i + 1]) for i in range(n - 2)]
    ax[0].plot(t[2:], fit_vals, "s--", ms=4, label=name.split(" ")[0] + " (навчання)")
ax[0].set(title="Відновлення ряду за всією вибіркою", xlabel="№ точки", ylabel="F")
ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
ax[1].plot(t, F, "o-", color="#c0392b", lw=2, label="Дійсні значення")
for name, idx in MODELS.items():
    if m - 2 < len(idx): continue
    a, *_ = fit(F[:m], idx)
    pr = forecast_recursive(F[:m], a, idx, 2)
    ax[1].plot([m, m + 1, m + 2], [F[m - 1]] + pr, "x--", label=name.split(" ")[0])
ax[1].axvline(m + .5, color="gray", ls=":")
ax[1].set(title="Прогноз 2 останніх точок (навчання за n−2)", xlabel="№ точки", ylabel="F")
ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig_results.png", dpi=150)
print("ok")