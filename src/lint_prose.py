"""Flag sentences in the rendered manuscript that contradict their own numbers.

check_claims.py verifies *conclusions* against the data. It cannot see a
sentence such as "outperforming the best baseline by -1.32 points", which is
what happened when a new baseline silently changed what a macro meant. This
scans the PDF text for phrasing that asserts a gain, lead or improvement and
then prints a negative number beside it.

    python src/lint_prose.py            # exit 0 = nothing suspicious
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF = os.path.join(ROOT, "paper", "main_en.pdf")

# a positive-direction verb/noun ... followed within a short window by "by -N"
GAIN_WORDS = (r"(outperform\w*|ahead of|higher than|above|improv\w*|gain\w*|"
              r"lift\w*|beat\w*|exceed\w*|surpass\w*|better than|stronger)")
NEG_NUM = r"by\s+[−\-–]\s?\d"
PATTERNS = [
    (re.compile(GAIN_WORDS + r"[^.;]{0,70}" + NEG_NUM, re.I),
     "positive-direction claim paired with a negative number"),
    (re.compile(r"(highest|best|strongest)[^.;]{0,40}\b[−\-–]\d+\.\d+ points", re.I),
     "superlative next to a negative margin"),
    (re.compile(r"\b(wins?|winning)\s+0/\d\s+seeds", re.I),
     "claims a win with zero seeds won"),
]


def main():
    if not os.path.exists(PDF):
        print("no PDF at", PDF)
        return 1
    txt = subprocess.run(["pdftotext", PDF, "-"], capture_output=True,
                         text=True).stdout.replace("\n", " ")
    hits = 0
    for pat, why in PATTERNS:
        for m in pat.finditer(txt):
            hits += 1
            start = max(0, m.start() - 40)
            print(f"  [{why}]\n    ...{txt[start:m.end() + 20]}...")
    print(f"\n{hits} suspicious sentence(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
