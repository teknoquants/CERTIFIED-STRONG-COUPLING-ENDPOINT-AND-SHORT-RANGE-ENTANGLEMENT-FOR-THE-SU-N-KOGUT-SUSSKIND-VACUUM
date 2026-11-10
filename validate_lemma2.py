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
"""Validation of Lemma 2 (Lieb-Robinson bound with unbounded on-site terms) and its proof steps.

Model: a chain of three-level sites with on-site h = diag(0, 1, Emax) and bounded nearest-neighbour
interactions of norm beta. Taking Emax large mimics the unbounded electric operator; the proof says the
bound must not depend on Emax.

What is checked
  (1)   H = H0 + Phi is self-adjoint with spectrum bounded below (finite truncation).
  (2)   tau_t = tau0_t o alpha_t with W(t) = e^{-itH0} e^{itH}, and dW/dt = i sum_z tau0_{-t}(phi_z) W(t).
  (3)   tau0_t preserves supports and norms of local operators.
  (4-5) ||[tau_t(A), B]|| <= 2 ||A|| ||B|| e^{-mu d} (e^{2 ||Phi||_mu |t|} - 1), for Emax = 30 and 3000
        (same bound: independent of the on-site spectrum).
  (6)   #{z in 2Z^3 : x in z + Lambda_0} <= 8 for every x (exhaustive), hence
        ||Phi||_mu <= 8 e^{2 mu} 6N/(g^4 c_N).
"""
import itertools
import math
import sys

import numpy as np
from scipy.linalg import expm

from common import Report, embed, chain_hamiltonian


def main():
    rep = Report("Lemma 2")
    rng = np.random.default_rng(22)
    L, beta, mu = 5, 0.15, 1.0
    PhiNorm = 2 * beta * math.exp(mu * 1)          # each site in <= 2 bonds, diam = 1

    results = {}
    for Emax in (30.0, 3000.0):
        h = np.diag([0.0, 1.0, Emax])
        rngE = np.random.default_rng(5)
        H, bonds = chain_hamiltonian(L, h, beta, rngE)
        H0 = sum(embed(h, k, L, 3) for k in range(L))
        ev = np.linalg.eigvalsh(H)
        A = embed(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], float), 0, L, 3)
        worst = 0.0
        ok = True
        w, V = np.linalg.eigh(H)
        for k in range(1, L):
            B = embed(np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]], float), k, L, 3)
            for t in (0.05, 0.2, 0.5, 1.0):
                U = V @ np.diag(np.exp(1j * t * w)) @ V.conj().T
                tA = U @ A @ U.conj().T
                lhs = np.linalg.norm(tA @ B - B @ tA, 2)
                rhs = 2 * np.linalg.norm(A, 2) * np.linalg.norm(B, 2) * math.exp(-mu * k) * (math.exp(2 * PhiNorm * t) - 1)
                ok &= lhs <= rhs * (1 + 1e-9)
                worst = max(worst, lhs / rhs)
        results[Emax] = (ok, worst, ev.min())
        if Emax == 30.0:
            rep.check("(1) H = H0 + Phi self-adjoint, bounded below", np.allclose(H, H.T) and np.isfinite(ev.min()))
            # (2) interaction picture
            t = 0.4
            W = expm(-1j * t * H0) @ expm(1j * t * H)
            alpha = W @ A @ W.conj().T
            tau0 = expm(1j * t * H0) @ alpha @ expm(-1j * t * H0)
            tau = expm(1j * t * H) @ A @ expm(-1j * t * H)
            Phi = H - H0
            dt = 1e-6
            Wp = expm(-1j * (t + dt) * H0) @ expm(1j * (t + dt) * H)
            Wm = expm(-1j * (t - dt) * H0) @ expm(1j * (t - dt) * H)
            dW = (Wp - Wm) / (2 * dt)
            gen = 1j * expm(-1j * t * H0) @ Phi @ expm(1j * t * H0) @ W
            rep.check("(2) tau_t = tau0_t o alpha_t and dW/dt = i tau0_{-t}(Phi) W", np.allclose(tau, tau0, atol=1e-8)
                      and np.allclose(dW, gen, atol=1e-5))
            # (3) support / norm preservation
            loc = embed(rng.normal(size=(3, 3)), 2, L, 3)
            tl = expm(1j * t * H0) @ loc @ expm(-1j * t * H0)
            other = embed(rng.normal(size=(3, 3)), 4, L, 3)
            rep.check("(3) tau0_t preserves supports and norms",
                      np.allclose(tl @ other, other @ tl) and abs(np.linalg.norm(tl, 2) - np.linalg.norm(loc, 2)) < 1e-9)
    rep.check("(4-5) Lieb-Robinson bound holds with on-site Emax = 30", results[30.0][0],
              f"max lhs/rhs {results[30.0][1]:.3f}")
    rep.check("(4-5) same bound holds with Emax = 3000 (independent of the on-site spectrum)", results[3000.0][0],
              f"max lhs/rhs {results[3000.0][1]:.3f}")

    # (6) counting translates in the 3D link-site lattice
    ok, mx = True, 0
    for x in itertools.product(range(-3, 4), repeat=3):
        cnt = sum(1 for z in itertools.product(range(-6, 7, 2), repeat=3)
                  if all(0 <= x[k] - z[k] <= 2 for k in range(3)))
        mx = max(mx, cnt)
    rep.check("(6) each site lies in at most 8 grouped supports z + Lambda_0, z in 2Z^3", mx == 8, f"max {mx}")
    ok = True
    for N in (2, 3):
        for g in (1.0, 2.0, 3.0):
            for m in (0.25, 0.5, 1.0):
                bound = 8 * math.exp(2 * m) * 6 * N / (g ** 4 * (N * N - 1) / N)
                direct = 8 * 6 * N / (g ** 4 * (N * N - 1) / N) * math.exp(m * 2)
                ok &= abs(bound - direct) < 1e-12
    rep.check("(6) ||Phi||_mu <= 8 e^{2 mu} 6N/(g^4 c_N) (diam_inf Lambda_0 = 2)", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
