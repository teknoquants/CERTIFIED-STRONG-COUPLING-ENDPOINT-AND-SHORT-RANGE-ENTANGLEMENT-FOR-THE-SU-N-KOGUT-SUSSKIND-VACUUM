# MIT License
#
# Copyright (c) 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Validation of the Definitions (Section 0) of the B8 white paper.

What is checked
  D1   c_N = (N^2-1)/N is the smallest positive eigenvalue of -Laplacian on SU(N) (metric -Tr):
       Casimir <lambda, lambda + 2 rho> over all dominant weights with labels <= 2, N = 2..6.
  D4   H_L(g) = (g^2 c_N / 2) Hbar_L + g^{-2} N |P_L| 1, term by term (symbolic, SymPy).
  D8   beta* = 1/kappa_Y = min{c1/2, 1/(2 c2)}.
  D9   (6 N kappa_Y / c_N)^(1/4) = (6 N^2 kappa_Y / (N^2-1))^(1/4).
  D11  For SU(2): sup_{0<beta<=1} [ln Z(beta) - (D/2) ln(1/beta)] is finite (gives c_Z), with D = 3.
  D12  Strip size: |S_w(A)| <= (2w+1)^3 |dA| for random regions in a box (link-site embedding).
"""
import math
import sys

import numpy as np
import sympy as sp

from common import Report, c_N, kappa_Y, g0_kappa, Z_su2


def casimir_sun(lam, N):
    """<lambda, lambda + 2 rho> in fundamental-weight coordinates, long roots of length^2 2."""
    r = N - 1
    Ainv = np.array([[min(i, j) - i * j / N for j in range(1, r + 1)] for i in range(1, r + 1)])
    lam = np.array(lam, dtype=float)
    return float(lam @ Ainv @ (lam + 2.0))


def c_Z_su2():
    betas = np.geomspace(1e-4, 1.0, 400)
    vals = [math.log(Z_su2(b, nmax=4000)) - 1.5 * math.log(1 / b) for b in betas]
    return max(vals), vals


def main():
    rep = Report("Definitions")
    rng = np.random.default_rng(1)

    # D1
    ok, det = True, []
    import itertools
    for N in range(2, 7):
        best = min(casimir_sun(l, N) for l in itertools.product(range(3), repeat=N - 1) if any(l))
        ok &= abs(best - c_N(N)) < 1e-12
        det.append(f"N={N}: {best:.4f}")
    rep.check("D1: c_N = (N^2-1)/N is the smallest positive Casimir", ok, "; ".join(det))

    # D4
    g, N, cN, P = sp.symbols("g N c_N P", positive=True)
    Lap, R = sp.symbols("Lap R")          # Lap = sum(-Delta_e), R = sum ReTr rho(U_p)
    h = Lap / cN
    lam = g ** -4
    Hbar = h + lam * (-(2 / cN) * R)
    lhs = (g ** 2 / 2) * Lap + g ** -2 * (N * P - R)
    rhs = (g ** 2 * cN / 2) * Hbar + g ** -2 * N * P
    rep.check("D4: H_L(g) = (g^2 c_N/2) Hbar_L + g^{-2} N |P_L| 1", sp.simplify(lhs - rhs) == 0)

    # D8, D9
    ok = True
    for _ in range(200):
        c1, c2 = rng.uniform(0.01, 2), rng.uniform(0.1, 10)
        kY = kappa_Y(c1, c2)
        ok &= abs(1 / kY - min(c1 / 2, 1 / (2 * c2))) < 1e-12
        for Nn in (2, 3, 5):
            ok &= abs(g0_kappa(Nn, kY) - (6 * Nn ** 2 * kY / (Nn ** 2 - 1)) ** 0.25) < 1e-12
    rep.check("D8, D9: beta* = 1/kappa_Y and the two forms of g0(N) agree", ok)

    # D11
    cZ, vals = c_Z_su2()
    tail = vals[:40]
    rep.check("D11: ln Z(beta) - (3/2) ln(1/beta) is bounded on (0,1] (SU(2)); c_Z finite",
              np.isfinite(cZ) and max(tail) - min(tail) < 0.05, f"c_Z = {cZ:.4f}")

    # D12
    ok = True
    box = 5
    links = [(np.array(x), i) for x in np.ndindex(box + 1, box + 1, box + 1) for i in range(3)
             if np.array(x)[i] + 1 <= box]
    sites = np.array([2 * x + np.eye(3, dtype=int)[i] for x, i in links])
    for trial in range(20):
        maskA = rng.random(len(sites)) < rng.uniform(0.2, 0.6)
        if maskA.all() or not maskA.any():
            continue
        A, Ac = sites[maskA], sites[~maskA]
        dist = np.max(np.abs(A[:, None, :] - Ac[None, :, :]), axis=2)
        dA = A[dist.min(axis=1) <= 2]
        for w in (0, 1, 2):
            dd = np.max(np.abs(sites[:, None, :] - dA[None, :, :]), axis=2).min(axis=1)
            Sw = int((dd <= w).sum())
            ok &= Sw <= (2 * w + 1) ** 3 * len(dA)
    rep.check("D12: |S_w(A)| <= (2w+1)^3 |dA| on random regions", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
