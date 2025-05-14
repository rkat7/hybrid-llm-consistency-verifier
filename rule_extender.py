

import os
import glob
import openai


export OPENAI_API_KEY=""

def extract_rules_from_corpus(corpus_dir: str, output_file: str, model: str = "gpt-4"):
    text_snippets = []
    for filepath in glob.glob(os.path.join(corpus_dir, '*.txt')):
        with open(filepath, 'r', encoding='utf-8') as f:
            text_snippets.append(f.read())
    corpus_text = "\n".join(text_snippets)

    prompt = (
        "Given the following text corpus, extract a concise set of Answer Set Programming (ASP) "
        "rules that capture the key relations, constraints, and domain facts implied in the text. "
        "Output each rule on its own line, using ASP syntax (e.g., 'married(john).', 'not_single(mary).').\n\n"
        f"Corpus:\n{corpus_text}\n\nRules:\n"
    )

    resp = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in logical rule extraction."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=1500
    )

    rules_text = resp.choices[0].message.content.strip()

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("% Auto-generated rules from corpus\n")
        out.write(rules_text)
        out.write("\n")

    print(f"[llm_rule_extender] Wrote {len(rules_text.splitlines())} rules to {output_file}")
