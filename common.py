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
"""Shared constants, SU(2) tools and a small check recorder.

Companion code to "Certified Strong-Coupling Endpoint and Short-Range Entanglement for the
SU(N) Kogut-Susskind Vacuum" (R. Cooperman and A.E. Petrunina V.V., XISOSIS, Vienna, 10 November 2026).
"""
import math
import numpy as np

# ---------------------------------------------------------------- constants
def c_N(N):
    """Smallest positive eigenvalue of -Laplacian on SU(N), metric -Tr(XY)."""
    return (N * N - 1) / N


def b0(N):
    return 11 * N / (48 * math.pi ** 2)


def b1(N):
    return 17 * N ** 2 / (384 * math.pi ** 4)


def a_two_loop(g, N, Lam=1.0):
    """Stipulated two-loop scale a(g) = Lam^-1 (b0 g^2)^(-b1/(2 b0^2)) exp(-1/(2 b0 g^2))."""
    B0, B1 = b0(N), b1(N)
    return (B0 * g * g) ** (-B1 / (2 * B0 * B0)) * math.exp(-1.0 / (2 * B0 * g * g)) / Lam


def a_one_loop(g, N, Lam=1.0):
    return math.exp(-1.0 / (2 * b0(N) * g * g)) / Lam


def beta_star(c1, c2):
    return min(c1 / 2.0, 1.0 / (2.0 * c2))


def g0(N, c1, c2):
    return (6 * N / (c_N(N) * beta_star(c1, c2))) ** 0.25


# ---------------------------------------------------------------- SU(2) tools
def haar_su2(n, rng):
    """n Haar-random SU(2) matrices, from uniform unit quaternions."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    a, b, c, d = q.T
    U = np.empty((n, 2, 2), dtype=complex)
    U[:, 0, 0] = a + 1j * b
    U[:, 0, 1] = c + 1j * d
    U[:, 1, 0] = -c + 1j * d
    U[:, 1, 1] = a - 1j * b
    return U


def su2_casimir(n):
    """-Laplacian eigenvalue on the spin-j irrep, n = 2j+1, metric -Tr: (n^2-1)/2."""
    return (n * n - 1) / 2.0


def heat_kernel_su2(theta, t, nmax=200):
    """K_t as a class function of theta, U ~ diag(e^{i theta}, e^{-i theta}).

    K_t = sum_n n chi_n e^{-t C_n / 2}, chi_n(theta) = sin(n theta)/sin(theta).
    """
    theta = np.asarray(theta, dtype=float)
    s = np.sin(theta)
    s = np.where(np.abs(s) < 1e-12, 1e-12, s)
    out = np.zeros_like(theta)
    for n in range(1, nmax + 1):
        w = math.exp(-t * su2_casimir(n) / 2.0)
        if w < 1e-300:
            break
        out += n * np.sin(n * theta) / s * w
    return out


def sample_heat_kernel_su2(n, t, rng, grid=20001):
    """Draw n SU(2) elements with density K_t relative to Haar (conjugation invariant)."""
    th = np.linspace(0.0, math.pi, grid)
    dens = heat_kernel_su2(th, t) * (2 / math.pi) * np.sin(th) ** 2
    dens = np.clip(dens, 0, None)
    cdf = np.cumsum(dens)
    cdf /= cdf[-1]
    theta = np.interp(rng.random(n), cdf, th)
    V = haar_su2(n, rng)
    D = np.zeros((n, 2, 2), dtype=complex)
    D[:, 0, 0] = np.exp(1j * theta)
    D[:, 1, 1] = np.exp(-1j * theta)
    return V @ D @ np.conj(np.transpose(V, (0, 2, 1)))


def chi_su2(U, n):
    """Character of the n-dimensional irrep evaluated on a batch of SU(2) matrices."""
    half_tr = np.real(np.trace(U, axis1=1, axis2=2)) / 2.0
    theta = np.arccos(np.clip(half_tr, -1.0, 1.0))
    s = np.sin(theta)
    small = np.abs(s) < 1e-9
    val = np.sin(n * theta) / np.where(small, 1.0, s)
    # limits: chi = n at theta = 0 and n(-1)^(n-1) at theta = pi
    val = np.where(small & (half_tr > 0), n, val)
    val = np.where(small & (half_tr < 0), n * (-1) ** (n - 1), val)
    return val


# ---------------------------------------------------------------- B8 helpers
def kappa_Y(c1, c2):
    return max(2.0 / c1, 2.0 * c2)


def g0_kappa(N, kY):
    """g0(N) = (6 N kappa_Y / c_N)^(1/4)  (Definition 9)."""
    return (6 * N * kY / c_N(N)) ** 0.25


def su2_h_spectrum(nmax):
    """Normalized link operator h = -Laplacian/c_2 on L^2(SU(2)): eigenvalue (n^2-1)/3, multiplicity n^2."""
    ev, mult = [], []
    for n in range(1, nmax + 1):
        ev.append((n * n - 1) / 3.0)
        mult.append(n * n)
    return np.array(ev), np.array(mult)


def Z_su2(beta, nmax=600):
    ev, mult = su2_h_spectrum(nmax)
    return float(np.sum(mult * np.exp(-beta * ev)))


def haar_sun(n, N, rng):
    """n Haar-random SU(N) matrices via QR of complex Ginibre matrices."""
    X = (rng.normal(size=(n, N, N)) + 1j * rng.normal(size=(n, N, N))) / np.sqrt(2)
    Q, R = np.linalg.qr(X)
    d = np.diagonal(R, axis1=1, axis2=2)
    Q = Q * (d / np.abs(d))[:, None, :]
    det = np.linalg.det(Q)
    Q[:, :, 0] /= det[:, None]
    return Q


def embed(op, site, L, d):
    return np.kron(np.kron(np.eye(d ** site), op), np.eye(d ** (L - site - 1)))


def embed2(op, site, L, d):
    return np.kron(np.kron(np.eye(d ** site), op), np.eye(d ** (L - site - 2)))


def chain_hamiltonian(L, h, beta, rng):
    """H = sum_k h_k + sum_k X_{k,k+1}, X real symmetric with ||X|| = beta (seeded)."""
    d = h.shape[0]
    H = sum(embed(h, k, L, d) for k in range(L))
    bonds = []
    for k in range(L - 1):
        X = rng.normal(size=(d * d, d * d)); X = X + X.T
        X *= beta / np.linalg.norm(X, 2)
        bonds.append(X)
        H = H + embed2(X, k, L, d)
    return H, bonds


def reduced_state(psi, keep, L, d):
    psi = psi.reshape([d] * L)
    rest = [k for k in range(L) if k not in keep]
    perm = list(keep) + rest
    M = np.transpose(psi, perm).reshape(d ** len(keep), -1)
    return M @ M.conj().T


def vn_entropy(rho):
    w = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    w = w[w > 1e-15]
    return float(-np.sum(w * np.log(w)))


def binary_entropy(e):
    if e <= 0 or e >= 1:
        return 0.0
    return -e * math.log(e) - (1 - e) * math.log(1 - e)


# ---------------------------------------------------------------- reporting
class Report:
    def __init__(self, name):
        self.name = name
        self.rows = []

    def check(self, label, ok, detail=""):
        self.rows.append((label, bool(ok), detail))
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
        return ok

    def summary(self):
        n = len(self.rows)
        p = sum(r[1] for r in self.rows)
        print(f"  -> {self.name}: {p}/{n} checks passed")
        return p == n
