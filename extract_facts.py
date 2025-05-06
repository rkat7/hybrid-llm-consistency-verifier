#!/usr/bin/env python3
import sys
import spacy

nlp = spacy.load("en_core_web_sm")

def sentence_to_fact(sent):
    """
    Very simple pattern-based extractor:
      • "X is Y."      →  y(x).
      • "X cannot Y."  →  not_y(x).
      • fallback: ignore.
    """
    doc = nlp(sent)
    subj = None
    pred = None
    neg  = False

    for tok in doc:
        if tok.dep_ == "nsubj":
            subj = tok.lemma_.lower()
        if tok.lemma_ in ("be",) and tok.dep_ == "ROOT":
            # look for complement (attr or acomp)
            for child in tok.children:
                if child.dep_ in ("attr","acomp"):
                    pred = child.lemma_.lower()
        if tok.dep_ in ("neg",) or tok.lemma_.lower() in ("cannot",):
            neg = True

    if subj and pred:
        fact = f"{pred}({subj})."
        return ("not_" + pred + f"({subj}).") if neg else fact

    return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_facts.py input.txt facts.lp")
        sys.exit(1)

    with open(sys.argv[1]) as fin, open(sys.argv[2], "w") as fout:
        for line in fin:
            line = line.strip()
            if not line: continue
            fact = sentence_to_fact(line)
            if fact:
                fout.write(fact + "\n")

if __name__ == "__main__":
    main()
