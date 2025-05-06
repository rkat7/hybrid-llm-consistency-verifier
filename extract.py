# extract.py
import spacy

nlp = spacy.load("en_core_web_sm")

def sentence_to_fact(sent: str) -> str | None:
    doc = nlp(sent)
    subj = pred = None
    neg = False

    for tok in doc:
        if tok.dep_ == "nsubj":
            subj = tok.lemma_.lower()
        if tok.lemma_ in ("be",) and tok.dep_ == "ROOT":
            for child in tok.children:
                if child.dep_ in ("attr","acomp"):
                    pred = child.lemma_.lower()
        if tok.dep_ == "neg":
            neg = True

    if subj and pred:
        atom = f"{pred}({subj})."
        return f"not_{pred}({subj})." if neg else atom
    return None

def extract_facts(path: str) -> list[str]:
    facts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fact = sentence_to_fact(line.strip())
            if fact:
                facts.append(fact)
    return facts
