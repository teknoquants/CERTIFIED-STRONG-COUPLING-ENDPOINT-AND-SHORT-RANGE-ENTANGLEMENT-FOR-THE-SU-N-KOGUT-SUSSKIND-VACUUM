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
"""Validation of Theorem 5 (entanglement area law) and every step of its proof.

What is checked
  (i.0)  case eps > 1:  |A| < C|dA|  =>  |A| s(e4) < C |dA| s(e4).
  (i.1)  local unitaries V_A (x) V_B do not change S(sigma_A).
  (i.2)  W supported on the strip: S(Tr_{A^c} W Omega0) = S(reduced state of W Omega0 on S_w(A) cap A).
  (i.3)  Gibbs bound for the strip entropy (truncated SU(2) links).
  (i.4)  (2w+1)^3 <= (2 ell + 3)^3 (1 + ln|A|)^3 for w = ceil(ell ln|A|).
  (i.5)  (1/2)||rho_A - sigma_A||_1 <= (1/2)|| |psi><psi| - |phi><phi| ||_1 <= ||psi - phi||.
  (i.7)  Winter's energy-constrained continuity bound |S(rho) - S(sigma)| <= 2 eps S(gamma(E/eps)) + h(eps)
         on random state pairs of a truncated SU(2) link, and S(gamma(E')) <= beta E' + ln Z(beta).
  (i.8)  with beta = 1/|A|:  2 eps S(gamma(calE/eps)) <= 2 max{e4, E_A} + C|dA|(D ln|A| + 2 c_Z),
         using the exact SU(2) partition function, D = 3, |A| up to 10^6.
  (i.9)  explicit constant: S <= c5 |dA| (1 + ln|A|)^3 with
         c5 = (2 ell + 3)^3 s(E_A) + C (D + 2 c_Z) + 2 max{e4, E_A} + ln 2 + C s(e4).
  (ii)   along H(s) = H0 + s V on a gapped qubit chain: S(rho_A(0)) = 0 and S(rho_A(1)) <= int_0^1 |dS/ds| ds
         <= c_S |dA|; and S(rho_A) stays bounded as |A| grows (one-dimensional area law, illustration).

What is NOT checked: the hypotheses A_E and S themselves (open).
"""
import math
import sys

import numpy as np

from common import (Report, su2_h_spectrum, Z_su2, vn_entropy, binary_entropy, reduced_state,
                    chain_hamiltonian, embed)


def rand_unitary(n, rng):
    X = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    Q, R = np.linalg.qr(X)
    return Q * (np.diag(R) / np.abs(np.diag(R)))


def gibbs_entropy_at_energy(spec, Ep):
    """Max entropy at mean energy <= Ep for a finite spectrum (Gibbs state)."""
    if Ep >= spec.mean():
        return math.log(len(spec))
    lo, hi = 0.0, 1.0
    mean = lambda b: float(np.sum(spec * np.exp(-b * (spec - spec.min()))) / np.sum(np.exp(-b * (spec - spec.min()))))
    while mean(hi) > Ep:
        hi *= 2
        if hi > 1e8:
            return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if mean(mid) > Ep else (lo, mid)
    b = hi
    p = np.exp(-b * (spec - spec.min())); p /= p.sum()
    p = p[p > 1e-300]
    return float(-np.sum(p * np.log(p)))


def lnZ(beta):
    n = np.arange(1, 40001, dtype=float)
    return float(np.log(np.sum(n * n * np.exp(-beta * (n * n - 1) / 3.0))))


def main():
    rep = Report("Theorem 5")
    rng = np.random.default_rng(55)

    # c_Z for SU(2) over (0, 1]
    betas = np.geomspace(1e-6, 1.0, 300)
    cZ = max(lnZ(b) - 1.5 * math.log(1 / b) for b in betas)

    # (i.0)
    ok = True
    for _ in range(1000):
        C, dA, s4 = rng.uniform(0.1, 10), rng.integers(1, 50), rng.uniform(0.01, 5)
        A = rng.integers(1, int(C * dA) + 2)
        if A < C * dA:
            ok &= A * s4 < C * dA * s4
    rep.check("(i.0) eps > 1 => |A| s(e4) < C|dA| s(e4)", ok)

    # (i.1), (i.2): 6 qubits, A = {0,1,2}, strip {2,3}
    L = 6
    Om0 = np.zeros(2 ** L); Om0[0] = 1
    W = np.kron(np.kron(np.eye(4), rand_unitary(4, rng)), np.eye(4))
    VA, VB = rand_unitary(8, rng), rand_unitary(8, rng)
    sigma = np.kron(VA, VB) @ W @ Om0
    S_sigma = vn_entropy(reduced_state(sigma, [0, 1, 2], L, 2))
    S_W = vn_entropy(reduced_state(W @ Om0, [0, 1, 2], L, 2))
    S_strip = vn_entropy(reduced_state(W @ Om0, [2], L, 2))
    rep.check("(i.1) local unitaries leave S(sigma_A) unchanged", abs(S_sigma - S_W) < 1e-10, f"{S_sigma:.6f} = {S_W:.6f}")
    rep.check("(i.2) entropy reduces to the strip part S_w(A) cap A (<= ln dim)", abs(S_W - S_strip) < 1e-10
              and S_strip <= math.log(2) + 1e-12, f"{S_W:.6f} = {S_strip:.6f} <= ln 2")

    # (i.3) Gibbs bound for the strip, truncated SU(2) links
    ev, mult = su2_h_spectrum(3)
    spec = np.repeat(ev, mult)
    D1 = len(spec)
    ok = True
    for _ in range(30):
        psi = rng.normal(size=D1 * D1) + 1j * rng.normal(size=D1 * D1)
        psi /= np.linalg.norm(psi)
        rho = reduced_state(psi, [0], 2, D1)
        E = float(np.real(np.sum(np.diag(rho) * spec)))
        Strip = vn_entropy(rho)
        for b in (0.1, 0.5, 2.0):
            ok &= Strip <= b * E + math.log(np.sum(np.exp(-b * spec))) + 1e-9
    rep.check("(i.3) strip entropy <= |S cap A| (beta E + ln Z(beta)) (truncated SU(2))", ok)

    # (i.4)
    ok = True
    for ell in (0.5, 1.0, 3.0):
        for A in [2, 3, 10, 100, 10 ** 4, 10 ** 6]:
            w = math.ceil(ell * math.log(A))
            ok &= (2 * w + 1) ** 3 <= (2 * ell + 3) ** 3 * (1 + math.log(A)) ** 3 + 1e-9
    rep.check("(i.4) (2w+1)^3 <= (2 ell + 3)^3 (1 + ln|A|)^3, w = ceil(ell ln|A|)", ok)

    # (i.5)
    ok = True
    for _ in range(50):
        psi = rng.normal(size=64) + 1j * rng.normal(size=64); psi /= np.linalg.norm(psi)
        phi = psi + 0.2 * (rng.normal(size=64) + 1j * rng.normal(size=64)); phi /= np.linalg.norm(phi)
        phi *= np.exp(-1j * np.angle(np.vdot(psi, phi)))
        tdA = 0.5 * np.abs(np.linalg.eigvalsh(reduced_state(psi, [0, 1, 2], 6, 2) - reduced_state(phi, [0, 1, 2], 6, 2))).sum()
        tdF = 0.5 * np.abs(np.linalg.eigvalsh(np.outer(psi, psi.conj()) - np.outer(phi, phi.conj()))).sum()
        ok &= tdA <= tdF + 1e-12 and tdF <= np.linalg.norm(psi - phi) + 1e-12
    rep.check("(i.5) partial trace and pure-state trace distance <= vector distance", ok)

    # (i.7) Winter continuity on a truncated SU(2) link
    ev, mult = su2_h_spectrum(4)
    spec = np.repeat(ev, mult)
    Dd = len(spec)
    ok, worst = True, 0.0
    for _ in range(300):
        b1, b2 = rng.uniform(0.2, 3), rng.uniform(0.2, 3)
        def rnd(b):
            X = rng.normal(size=(Dd, Dd)) + 1j * rng.normal(size=(Dd, Dd))
            Dm = np.diag(np.exp(-0.5 * b * spec))
            r = Dm @ X @ X.conj().T @ Dm
            return r / np.trace(r).real
        rho = rnd(b1)
        t = rng.uniform(0, 0.5)
        sig = (1 - t) * rho + t * rnd(b2)
        eps = 0.5 * np.abs(np.linalg.eigvalsh(rho - sig)).sum()
        if eps < 1e-6:
            continue
        E = max(float(np.real(np.trace(rho @ np.diag(spec)))), float(np.real(np.trace(sig @ np.diag(spec)))))
        lhs = abs(vn_entropy(rho) - vn_entropy(sig))
        rhs = 2 * eps * gibbs_entropy_at_energy(spec, E / eps) + binary_entropy(eps)
        ok &= lhs <= rhs + 1e-9
        worst = max(worst, lhs / rhs)
    rep.check("(i.7) Winter: |S(rho) - S(sigma)| <= 2 eps S(gamma(E/eps)) + h(eps) (300 random pairs)", ok,
              f"max lhs/rhs {worst:.3f}")
    ok = True
    for Ep in (0.05, 0.5, 2.0):
        Sg = gibbs_entropy_at_energy(spec, Ep)
        for b in (0.1, 1.0, 5.0):
            ok &= Sg <= b * Ep + math.log(np.sum(np.exp(-b * spec))) + 1e-9
    rep.check("(i.7) S(gamma(E')) <= beta E' + ln Z(beta)", ok)

    # (i.8)
    ok, D = True, 3
    for A in [2, 10, 1000, 10 ** 6]:
        for C in (0.5, 2.0):
            for dA in (1, 5, 50):
                eps = C * dA / A
                if eps > 1:
                    continue
                for e4, EA in [(0.01, 0.1), (0.5, 0.2)]:
                    calE = A * max(e4, EA)
                    lhs = 2 * eps * ((1 / A) * calE / eps + A * lnZ(1 / A))
                    rhs = 2 * max(e4, EA) + C * dA * (D * math.log(A) + 2 * cZ)
                    ok &= lhs <= rhs * (1 + 1e-9)
    rep.check("(i.8) beta = 1/|A| gives 2 eps S(gamma) <= 2max{e4,E_A} + C|dA|(D ln|A| + 2 c_Z) (exact SU(2) Z)", ok,
              f"c_Z = {cZ:.4f}")

    # (i.9) explicit c5
    ok = True
    for _ in range(2000):
        ell, C, sE, s4, e4, EA = rng.uniform(0.2, 3), rng.uniform(0.1, 5), rng.uniform(0.01, 3), rng.uniform(0.01, 3), rng.uniform(0, 1), rng.uniform(0, 1)
        A, dA = int(rng.integers(2, 10 ** 6)), int(rng.integers(1, 1000))
        w = math.ceil(ell * math.log(A))
        bound = (2 * w + 1) ** 3 * dA * sE + C * dA * (D * math.log(A) + 2 * cZ) + 2 * max(e4, EA) + math.log(2) + C * dA * s4
        c5 = (2 * ell + 3) ** 3 * sE + C * (D + 2 * cZ) + 2 * max(e4, EA) + math.log(2) + C * s4
        ok &= bound <= c5 * dA * (1 + math.log(A)) ** 3 * (1 + 1e-12)
    rep.check("(i.9) S <= c5 |dA| (1+ln|A|)^3 with c5 = (2l+3)^3 s(E_A) + C(D+2c_Z) + 2max + ln2 + C s(e4)", ok)

    # (ii) path entropy and 1D area law
    L = 10
    h = np.diag([0.0, 1.0])
    rngc = np.random.default_rng(9)
    Hfull, bonds = chain_hamiltonian(L, h, 0.3, rngc)
    H0 = sum(embed(h, k, L, 2) for k in range(L))
    Vint = Hfull - H0
    ss = np.linspace(0, 1, 41)
    Ss = []
    keep = list(range(L // 2))
    for s in ss:
        w, Vv = np.linalg.eigh(H0 + s * Vint)
        Ss.append(vn_entropy(reduced_state(Vv[:, 0], keep, L, 2)))
    Ss = np.array(Ss)
    dS = np.abs(np.diff(Ss)) / np.diff(ss)
    integral = float(np.sum(np.abs(np.diff(Ss))))
    cS = float(dS.max())
    rep.check("(ii) S(rho_A(0)) = 0 and S(rho_A(1)) <= int |dS/ds| ds <= c_S |dA| (|dA| = 1)",
              Ss[0] < 1e-10 and Ss[-1] <= integral + 1e-12 and integral <= cS + 1e-12,
              f"S(1) = {Ss[-1]:.4f}, integral {integral:.4f}, c_S = {cS:.4f}")
    w, Vv = np.linalg.eigh(Hfull)
    ent = [vn_entropy(reduced_state(Vv[:, 0], list(range(k)), L, 2)) for k in range(1, L)]
    rep.check("(ii) illustration: S(rho_A) bounded uniformly in |A| (gapped 1D chain)", max(ent) < 0.5,
              "S by |A|: " + ", ".join(f"{x:.3f}" for x in ent))
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
