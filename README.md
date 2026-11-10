# Validation scripts: Certified Strong-Coupling Endpoint and Short-Range Entanglement (B8)

Companion code to the B8 white paper by R. Cooperman and A.E. Petrunina V.V. (XISOSIS, Vienna, 10 November 2026).
Released under the MIT License (see `LICENSE`; the licence is also reproduced at the top of every script).

GitHub: [GITHUB REPOSITORY URL — placeholder] · DOI: [DOI — placeholder]

## Run

```
pip install numpy scipy sympy
python run_all.py
```

Python 3.10+, NumPy ≥ 2.0, SciPy ≥ 1.9, SymPy. Runtime is about 30 seconds.

## Coverage

| Script | Paper item | Checks |
|---|---|---|
| `validate_definitions.py` | Definitions 1–14 | c_N is the smallest positive Casimir (SU(2)–SU(6)); the H_L ↔ H̄_L identity (symbolic); κ_Y, β*, g₀(N) relations; heat-kernel constant c_Z for SU(2); strip-size bound |
| `validate_theorem1.py` | Theorem 1, all proof steps | Injectivity and supports of the link-site embedding; sup\|ReTr\| = N; ≤ 3 plaquettes per corner; threshold equivalence; Yarotsky-type gap illustration; decoupling; rescaling; gauge invariance of the positive ground state; exact arithmetic for g₀(2), g₀(3), monotonicity and limit |
| `validate_lemma2.py` | Lemma 2, all proof steps | Interaction-picture identity and generator; support/norm preservation; the Lieb–Robinson bound with on-site level 30 and 3000 (bound independent of the electric spectrum); at most 8 grouped supports per site |
| `validate_theorem3.py` | Theorem 3, all proof ingredients | Lieb–Robinson input; gap; centring; Gaussian smoothing norm bound with an on-site level at 500; the clustering bound on qubit and three-level chains; explicit μ₃ at μ = ½ |
| `validate_lemma4.py` | Lemma 4, all proof steps | Frozen-link fibres, trial-state bound, spread and derivative bounds on a three-rotor chain; e₄ decomposition; Gibbs entropy bound on truncated SU(2) links |
| `validate_theorem5.py` | Theorem 5, all proof steps | Case ε > 1; local-unitary invariance and strip reduction; Winter's continuity bound (300 random pairs); the β = 1/\|A\| estimate with the exact SU(2) partition function up to \|A\| = 10⁶; an **explicit constant c₅**; part (ii) along a gapped path |

## What is not validated

The numerical value of κ_Y (Theorem 1's single open input), a line-by-line replay of the Nachtergaele–Sims contour argument used in Theorem 3, and the hypotheses 𝔄_E and 𝔖 of Theorem 5. Steps that concern the infinite three-dimensional lattice are checked on finite systems with the same operator structure (unbounded-like on-site spectra, bounded local interactions), not on SU(N) lattice gauge theory itself.

Byproduct: `validate_theorem5.py` supplies an explicit constant for Theorem 5(i),
c₅ = (2ℓ+3)³ s(E_𝔄) + C_𝔄(D + 2c_Z) + 2max{e₄, E_𝔄} + ln 2 + C_𝔄 s(e₄), with c_Z ≈ 1.1673 for SU(2).
