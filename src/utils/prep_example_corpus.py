# prep_example_corpus.py
import os
import nltk
from nltk.corpus import gutenberg, reuters

# 1. Download corpora (only once)
nltk.download("gutenberg")
nltk.download("reuters")

# 2. Ensure your corpus folder exists
CORPUS_DIR = "corpus"
os.makedirs(CORPUS_DIR, exist_ok=True)

# 3. Export a few Gutenberg texts
for fid in gutenberg.fileids():
    out = os.path.join(CORPUS_DIR, fid)        # e.g. 'corpus/melville-moby_dick.txt'
    with open(out, "w", encoding="utf-8") as f:
        f.write(gutenberg.raw(fid))

# 4. Export a handful of Reuters categories (for brevity, pick just 'trade' & 'crude')
for cat in ("trade", "crude"):
    paths = reuters.fileids(cat)
    for pid in paths[:50]:                      # take first 50 articles per category
        out = os.path.join(CORPUS_DIR, pid.replace("/", "_") + ".txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(reuters.raw(pid))

print(f"Wrote {len(os.listdir(CORPUS_DIR))} files into {CORPUS_DIR}/")
