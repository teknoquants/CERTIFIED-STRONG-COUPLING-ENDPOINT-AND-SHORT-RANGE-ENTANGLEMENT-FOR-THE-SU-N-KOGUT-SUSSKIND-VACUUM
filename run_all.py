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
"""Run every validation script for the B8 white paper
"Certified Strong-Coupling Endpoint and Short-Range Entanglement for the SU(N) Kogut-Susskind Vacuum".

Usage:  python run_all.py      (exit code 0 if every check passes)
"""
import sys
import time

import validate_definitions
import validate_theorem1
import validate_lemma2
import validate_theorem3
import validate_lemma4
import validate_theorem5

SUITES = [
    ("Definitions D1-D14", validate_definitions),
    ("Theorem 1 - certified strong-coupling endpoint", validate_theorem1),
    ("Lemma 2 - Lieb-Robinson bound", validate_lemma2),
    ("Theorem 3 - exponential clustering", validate_theorem3),
    ("Lemma 4 - local electric energy and Gibbs bound", validate_lemma4),
    ("Theorem 5 - entanglement area law", validate_theorem5),
]


def main():
    results = []
    for title, mod in SUITES:
        print(f"\n=== {title} ===")
        t = time.time()
        ok = mod.main()
        results.append((title, ok, time.time() - t))
    print("\n=== Summary ===")
    for title, ok, dt in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {title}  ({dt:.1f} s)")
    print("\nNot validated (open inputs of the paper): the numerical value of kappa_Y, a line-by-line replay of the\n"
          "Nachtergaele-Sims contour argument, and the hypotheses A_E and S of Theorem 5. Steps that concern the\n"
          "infinite lattice are checked on finite chains with the same operator structure.")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
