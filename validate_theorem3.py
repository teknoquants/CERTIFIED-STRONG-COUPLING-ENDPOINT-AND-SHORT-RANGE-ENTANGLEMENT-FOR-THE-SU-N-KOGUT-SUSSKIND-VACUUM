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
"""Validation of Theorem 3 (exponential clustering) and the ingredients of its proof.

What is checked
  (1)   Lieb-Robinson input in the form used (Lemma 2) on the test chain.
  (2)   Gap above a nondegenerate ground state (gamma >= 1/2) on the test chains.
  (3)   Centring: B~ = B - <B> leaves commutators unchanged, P0 B~ Omega = 0, <AB> - <A><B> = <A B~>.
  (4-5) Gaussian time average: z -> tau_z(B(a)) entire with ||tau_z(B(a))|| <= ||B|| e^{a (Im z)^2},
        computed exactly in the energy basis, also for an on-site spectrum with a level at 500
        (unbounded mimic): only real-time dynamics enters.
  (7)   The clustering bound |<AB> - <A><B>| <= ||A|| ||B|| (1 + (2/pi) min(|X|,|Y|) + (pi mu3 d)^{-1/2}) e^{-mu3 d}
        with mu3 = (mu/2)/(4 ||Phi||_mu + 1/2), on a 10-site qubit chain and a 6-site three-level chain.
  (8)   mu = 1/2:  mu3 >= (1/4)/(192 e N/(g^4 c_N) + 1/2) from 4||Phi||_{1/2} <= 192 e N/(g^4 c_N).

What is NOT checked: a line-by-line replay of [5]'s contour argument; the test chains only confirm that the
stated bound holds where all quantities are computable.
"""
import math
import sys

import numpy as np

from common import Report, embed, chain_hamiltonian, c_N


def clustering_check(L, h, beta, mu, rng):
    d = h.shape[0]
    H, _ = chain_hamiltonian(L, h, beta, rng)
    w, V = np.linalg.eigh(H)
    gap = w[1] - w[0]
    Om = V[:, 0]
    Phi = 2 * beta * math.exp(mu)
    mu3 = (0.5 * mu) / (4 * Phi + 0.5)
    ok, worst = True, 0.0
    a_op = np.zeros((d, d)); a_op[0, 1] = a_op[1, 0] = 1
    for k in range(1, L):
        A = embed(a_op, 0, L, d)
        B = embed(a_op, k, L, d)
        corr = Om @ A @ B @ Om - (Om @ A @ Om) * (Om @ B @ Om)
        rhs = (1 + 2 / math.pi + (math.pi * mu3 * k) ** -0.5) * math.exp(-mu3 * k)
        ok &= abs(corr) <= rhs
        worst = max(worst, abs(corr) / rhs)
    return gap, ok, worst, H, w, V


def main():
    rep = Report("Theorem 3")
    rng = np.random.default_rng(33)
    mu = 0.5

    gap_q, ok_q, worst_q, H, w, V = clustering_check(10, np.diag([0.0, 1.0]), 0.1, mu, rng)
    rep.check("(2) gap above a nondegenerate ground state, gamma >= 1/2 (qubit chain, L = 10)", gap_q >= 0.5,
              f"gap {gap_q:.3f}")

    # (1) LR input form on this chain (single site A at 0, B at k)
    Om = V[:, 0]
    a_op = np.array([[0, 1], [1, 0]], float)
    A = embed(a_op, 0, 10, 2)
    Phi = 2 * 0.1 * math.exp(mu)
    ok = True
    for k in (2, 4, 6):
        B = embed(a_op, k, 10, 2)
        for t in (0.3, 1.0):
            U = V @ np.diag(np.exp(1j * t * w)) @ V.T
            tA = U @ A @ U.conj().T
            ok &= np.linalg.norm(tA @ B - B @ tA, 2) <= 2 * math.exp(-mu * k) * math.exp(2 * Phi * t)
    rep.check("(1) Lieb-Robinson input: ||[tau_t(A),B]|| <= 2||A||||B|| e^{-mu d} e^{2||Phi|| t}", ok)

    # (3) centring
    B = embed(a_op, 5, 10, 2)
    Bt = B - (Om @ B @ Om) * np.eye(B.shape[0])
    P0 = np.outer(Om, Om)
    rep.check("(3) centring: [tau_t(A), B~] = [tau_t(A), B], P0 B~ Omega = 0, <AB> - <A><B> = <A B~>",
              np.allclose(A @ Bt - Bt @ A, A @ B - B @ A) and np.linalg.norm(P0 @ Bt @ Om) < 1e-12 and
              abs((Om @ A @ B @ Om - (Om @ A @ Om) * (Om @ B @ Om)) - Om @ A @ Bt @ Om) < 1e-12)

    # (4-5) Gaussian smoothing, exact in the energy basis, with a very high on-site level
    Lg = 5
    hg = np.diag([0.0, 1.0, 500.0])
    Hg, _ = chain_hamiltonian(Lg, hg, 0.1, np.random.default_rng(4))
    wg, Vg = np.linalg.eigh(Hg)
    Bg = embed(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], float), 2, Lg, 3)
    Bt_g = Vg.T @ Bg @ Vg
    omega = wg[:, None] - wg[None, :]
    ok, worst = True, 0.0
    for a in (0.1, 1.0, 5.0):
        for z in (0.0, 1.3 + 0.4j, -2 + 1.0j, 0.5 - 0.8j):
            # tau_z(B(a))_{mn} = B_mn exp(i z omega_mn - omega_mn^2 / (4a))
            M = Bt_g * np.exp(1j * z * omega - omega ** 2 / (4 * a))
            lhs = np.linalg.norm(M, 2)
            rhs = np.linalg.norm(Bg, 2) * math.exp(a * np.imag(z) ** 2)
            ok &= lhs <= rhs * (1 + 1e-9)
            worst = max(worst, lhs / rhs)
    rep.check("(4-5) ||tau_z(B(a))|| <= ||B|| e^{a (Im z)^2}, valid with an on-site level at 500", ok,
              f"max ratio {worst:.3f}")

    rep.check("(7) clustering bound with mu3 = (mu/2)/(4||Phi||_mu + 1/2): qubit chain L = 10", ok_q,
              f"max |corr|/bound {worst_q:.2e}")
    gap_t, ok_t, worst_t, *_ = clustering_check(6, np.diag([0.0, 1.0, 40.0]), 0.1, mu, np.random.default_rng(8))
    rep.check("(7) clustering bound: three-level chain L = 6 with on-site level at 40", ok_t and gap_t >= 0.5,
              f"gap {gap_t:.3f}, max |corr|/bound {worst_t:.2e}")

    # (8) explicit mu3 at mu = 1/2 for the 3D model
    ok = True
    for N in (2, 3):
        for g in np.linspace(1.0, 4.0, 13):
            beta = 6 * N / (g ** 4 * c_N(N))
            Phi_half = 8 * math.exp(1.0) * beta
            mu3 = 0.25 / (4 * Phi_half + 0.5)
            ok &= abs(4 * Phi_half - 192 * math.e * N / (g ** 4 * c_N(N))) < 1e-9
            ok &= mu3 >= 0.25 / (192 * math.e * N / (g ** 4 * c_N(N)) + 0.5) * (1 - 1e-12)
    rep.check("(8) mu = 1/2: 4||Phi||_{1/2} <= 192 e N/(g^4 c_N) and mu3 >= (1/4)/(192 e N/(g^4 c_N) + 1/2)", ok)
    return rep.summary()


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
