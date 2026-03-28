"""
Plots the per-dimension scaling factor Lambda_j for PI, NTK-aware, and YaRN
as a function of dimension index j, with a vertical line marking j_c.

Parameters chosen to match LLaMA-style models:
  theta = 10000, D_H = 128, S = 4096, S' = 16384 (4x extension)
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import os

# ── Parameters ────────────────────────────────────────────────────────────────
theta   = 10_000
D_H     = 128
S       = 4_096
S_prime = 16_384
s       = S_prime / S          # extension ratio = 4

N  = D_H // 2                  # number of dimension pairs (0 … N-1)
js = np.arange(N, dtype=float)

# ── Critical / boundary dimensions ────────────────────────────────────────────
j_c = (D_H / 2) * math.log(S       / (2 * math.pi)) / math.log(theta)
j_h = (D_H / 2) * math.log(S       / (64 * math.pi)) / math.log(theta)
j_l = j_c   # same as j_c (YaRN low-freq boundary = one full rotation)

# ── Scaling factors ────────────────────────────────────────────────────────────
# PI: uniform 1/s
lambda_pi = np.full(N, 1.0 / s)

# NTK-aware: alpha^{-2j/D_H},  alpha = s^{D_H / (D_H-2)}
alpha     = s ** (D_H / (D_H - 2))
lambda_ntk = alpha ** (-2 * js / D_H)

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(js, lambda_pi,   color='#4393c3', linewidth=2.0, linestyle='--',  label='PI')
ax.plot(js, lambda_ntk,  color='#d6604d', linewidth=2.0, linestyle='-.',  label='NTK-aware')

ax.axvline(x=j_c, color='#666', linestyle=':', linewidth=1.5,
           label=fr'$j_c = {j_c:.1f}$')

# reference lines at the two extremes
ax.axhline(y=1.0,   color='#ccc', linestyle=':', linewidth=1.0, zorder=0)
ax.axhline(y=1.0/s, color='#ccc', linestyle=':', linewidth=1.0, zorder=0)

# labels
ax.text(N - 1, 1.0 + 0.02,   r'$\Lambda=1$ (pure extrapolation)',
        ha='right', va='bottom', fontsize=9, color='#666')
ax.text(N - 1, 1.0/s + 0.02, r'$\Lambda=S/S^{\prime}$ (full compression)',
        ha='right', va='bottom', fontsize=9, color='#666')

ax.set_xlabel('Dimension index $j$', fontsize=12)
ax.set_ylabel(r'Scaling factor $\Lambda_j$', fontsize=12)
ax.set_title(
    fr'Per-dimension scaling: $\theta={theta}$, $S={S}$, $S^{{\prime}}={S_prime}$',
    fontsize=12
)
ax.set_xlim(0, N - 1)
ax.set_ylim(-0.04, 1.18)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

out = os.path.join(os.path.dirname(__file__), 'lambda_comparison.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved → {out}')
