# llm_rule_extender_with_corpus.py

from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import torch
import os
import random

def sample_corpus_sentences(corpus_dir, sample_size=20):
    """Randomly pick a handful of sentences from your .txt files."""
    sents = []
    for fn in os.listdir(corpus_dir):
        if not fn.endswith(".txt"): continue
        with open(os.path.join(corpus_dir, fn), encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines:
                # sample up to one sentence per document
                sents.append(random.choice(lines))
                if len(sents) >= sample_size:
                    break
    return sents

def extend_domain_rules_with_corpus(
    rules_path: str,
    corpus_dir: str,
    model_name: str = "EleutherAI/gpt2-medium",
    num_rules: int = 5
):
    # 1) Load existing rules
    with open(rules_path, "r", encoding="utf-8") as f:
        existing = f.read()

    # 2) Sample example sentences from your corpus
    examples = sample_corpus_sentences(corpus_dir)
    example_block = "\n".join(f"- “{s}”" for s in examples)

    # 3) Build a richer prompt
    prompt = (
        "You’re an ASP expert helping us catch logical inconsistencies in English text.\n\n"
        "Current ASP rules:\n```\n" + existing + "\n```\n\n"
        "Here are a few sample sentences from our corpus:\n"
        f"{example_block}\n\n"
        f"Based on these examples, propose {num_rules} new ASP constraints "
        "(show only code lines)."
    )

    # 4) Load model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    generator = TextGenerationPipeline(model, tokenizer, device=0 if torch.cuda.is_available() else -1)

    # 5) Generate
    out = generator(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]

    # 6) Isolate new rules
    new_rules = out[len(prompt):].strip()

    # 7) Append
    with open(rules_path, "a", encoding="utf-8") as f:
        f.write("\n\n% === LLM-generated (corpus-aware) rules ===\n")
        f.write(new_rules + "\n")

    print(f"Appended {num_rules} new rules to {rules_path}")

if __name__ == "__main__":
    extend_domain_rules_with_corpus(
        rules_path="domain_rules.lp",
        corpus_dir="corpus/",
        model_name="EleutherAI/gpt2-medium",
        num_rules=5
    )
