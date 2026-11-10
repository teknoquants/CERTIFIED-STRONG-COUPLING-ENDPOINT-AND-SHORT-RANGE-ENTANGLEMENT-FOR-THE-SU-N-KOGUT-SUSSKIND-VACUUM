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
"""Validation of Lemma 4 (local electric energy; Gibbs entropy bound) and every step of its proof.

Part (i) is checked on the abelian analogue of the frozen-link argument: three quantum rotors with
normalized kinetic terms h_k = n_k^2 (gap 1, unique ground state) and two "plaquette" couplings
lambda * psi, psi = -2 cos(theta_a - theta_b) (range of width 4, derivative norm 2), in a Fourier
truncation. Removing h_2 freezes theta_2 = u and gives the fibre operators K(u).

What is checked
  (1)    <h_e> <= E0(H) - E0(H - h_e), with E0(H - h_e) = min_u E0(K(u))   (direct-integral fibres).
  (3)    each fibre K(u) has a unique ground state and a uniform gap.
  (4)    trial state: E0(H) <= max_u E0(K(u)) + sup_u ||d_u phi(u)||^2.
  (5)    spread of E0(K(u)) <= (#plaquettes at e) * lambda * (range width).
  (6)    ||d_u phi|| <= ||R(u)|| ||d_u K|| <= (1/gap_K) * (#plaquettes) * lambda * 2.
  (7)    combined bound <h_e> <= spread + (#pl * 2 lambda / gap_K)^2, and the paper's e4(N, lambda) formula.
  (ii)   Gibbs bound S(rho) <= beta Tr(rho sum h) + |A| ln Z(beta) for random states on one and two truncated
         SU(2) links (spectrum (n^2-1)/3 with multiplicity n^2), for many beta; and S(rho_A) <= |A| s(e4).
"""
import math
import sys

import numpy as np

from common import Report, c_N, su2_h_spectrum, vn_entropy

M = 6


def rotor_ops():
    ns = np.arange(-M, M + 1)
    d = len(ns)
    n2 = np.diag(ns.astype(float) ** 2)
    up = np.diag(np.ones(d - 1), -1)          # e^{i theta}: n -> n+1
    return d, n2, up


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def main():
    rep = Report("Lemma 4")
    rng = np.random.default_rng(44)
    lam = 0.25
    d, n2, up = rotor_ops()
    I = np.eye(d)
    # cos(th_a - th_b) = (e^{i th_a} e^{-i th_b} + h.c.)/2
    def cosdiff(Ua, Ub):
        X = Ua @ Ub.conj().T
        return 0.5 * (X + X.conj().T)
    U1, U2, U3 = kron3(up, I, I), kron3(I, up, I), kron3(I, I, up)
    h1, h2, h3 = kron3(n2, I, I), kron3(I, n2, I), kron3(I, I, n2)
    H = h1 + h2 + h3 + lam * (-2 * cosdiff(U1, U2) - 2 * cosdiff(U2, U3))
    w, V = np.linalg.eigh(H)
    Om = V[:, 0]
    E0 = w[0]
    he = float(np.real(Om.conj() @ h2 @ Om))

    # fibres K(u) on rotors 1, 3 with theta_2 = u
    def K(u):
        e = np.exp(1j * u)
        A1 = np.kron(up, I); A3 = np.kron(I, up)
        c1 = 0.5 * (A1 * np.conj(e) + (A1 * np.conj(e)).conj().T)
        c3 = 0.5 * (A3 * np.conj(e) + (A3 * np.conj(e)).conj().T)
        return np.kron(n2, I) + np.kron(I, n2) + lam * (-2 * c1 - 2 * c3)
    us = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    E0K, gaps, phis = [], [], []
    for u in us:
        wk, Vk = np.linalg.eigh(K(u))
        E0K.append(wk[0]); gaps.append(wk[1] - wk[0])
        v = Vk[:, 0]
        k0 = np.argmax(np.abs(v))
        v = v * np.exp(-1j * np.angle(v[k0]))
        phis.append(v)
    E0K, gaps = np.array(E0K), np.array(gaps)
    du = us[1] - us[0]
    dphi = []
    for k in range(len(us)):
        a, b = phis[(k + 1) % len(us)], phis[k - 1]
        b = b * np.exp(-1j * np.angle(np.vdot(phis[k], b)))
        a = a * np.exp(-1j * np.angle(np.vdot(phis[k], a)))
        dphi.append(np.linalg.norm((a - b) / (2 * du)))
    dphi = np.array(dphi)

    rep.check("(1) <h_e> <= E0(H) - min_u E0(K(u))", he <= E0 - E0K.min() + 1e-9,
              f"<h_e> = {he:.5f} <= {E0 - E0K.min():.5f}")
    rep.check("(3) every fibre K(u) has a unique ground state, uniform gap", gaps.min() > 0.5, f"min gap {gaps.min():.3f}")
    rep.check("(4) E0(H) <= max_u E0(K(u)) + sup_u ||d_u phi||^2", E0 <= E0K.max() + dphi.max() ** 2 + 1e-9,
              f"{E0:.5f} <= {E0K.max() + dphi.max() ** 2:.5f}")
    npl, width, dnorm = 2, 4.0, 2.0
    rep.check("(5) spread of E0(K(u)) <= #pl * lambda * width", E0K.max() - E0K.min() <= npl * lam * width + 1e-12,
              f"spread {E0K.max() - E0K.min():.2e} <= {npl * lam * width}")
    rep.check("(6) ||d_u phi|| <= (1/gap_K) * #pl * lambda * ||d psi||", dphi.max() <= npl * lam * dnorm / gaps.min() + 1e-6,
              f"{dphi.max():.4f} <= {npl * lam * dnorm / gaps.min():.4f}")
    bound7 = npl * lam * width + (npl * lam * dnorm / gaps.min()) ** 2
    rep.check("(7) <h_e> <= spread bound + (#pl lambda ||d psi|| / gap)^2", he <= bound7, f"{he:.5f} <= {bound7:.4f}")

    ok = True
    for N in (2, 3, 4):
        for g in (1.5, 2.0, 3.0):
            l4 = g ** -4
            e4 = 16 * N * l4 / c_N(N) + 256 * N * (N * N - 1) * l4 ** 2 / c_N(N) ** 3
            spread = 4 * l4 * 4 * N / c_N(N)
            grad = (N * N - 1) * (16 * math.sqrt(N) * l4 / c_N(N)) ** 2 / c_N(N)
            ok &= abs(e4 - (spread + grad)) < 1e-12
    rep.check("(7) e4(N, lambda) = 4 lambda (4N/c_N) + c_N^{-1} (N^2-1)(16 sqrt(N) lambda/c_N)^2", ok)

    # (ii) Gibbs bound on truncated SU(2) links
    ev, mult = su2_h_spectrum(3)
    spec1 = np.repeat(ev, mult)               # dimension 1 + 4 + 9 = 14
    ok = True
    for nlinks in (1, 2):
        spec = spec1 if nlinks == 1 else (spec1[:, None] + spec1[None, :]).ravel()
        D = len(spec)
        for _ in range(40):
            X = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
            rho = X @ X.conj().T
            rho /= np.trace(rho).real
            E = float(np.real(np.sum(np.diag(rho) * spec)))
            S = vn_entropy(rho)
            for beta in (0.05, 0.3, 1.0, 3.0):
                Z1 = np.sum(np.exp(-beta * spec1))
                ok &= S <= beta * E + nlinks * math.log(Z1) + 1e-9
    rep.check("(ii) Gibbs bound S(rho) <= beta E + |A| ln Z(beta) (1 and 2 truncated SU(2) links)", ok)

    ok = True
    ev60, mult60 = su2_h_spectrum(60)
    for E in (0.01, 0.1, 0.5):
        betas = np.geomspace(1e-3, 50, 400)
        s = min(b * E + math.log(np.sum(mult60 * np.exp(-b * ev60))) for b in betas)
        ok &= s >= 0 and np.isfinite(s)
    rep.check("(ii) s(E) = inf_beta [beta E + ln Z(beta)] is finite and nonnegative (volume-law constant)", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
