'''
Simple first step to check the last layer attentioon weights with related pairs.
Here we checked for noun-verb relation. 
Cannot extrapolate this to large amount of text
'''


import torch
import spacy
from transformers import BertTokenizerFast, BertModel


# Load models
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
model.eval()


# Sample input
sentence = "The big red dog chased the scared little cat."
doc = nlp(sentence)


# Identify (adjective, noun) pairs from dependency parse
adj_noun_pairs = [(tok.text, tok.head.text) for tok in doc if tok.dep_ == "amod"]


print("Adjective-Noun Pairs:", adj_noun_pairs)


encoded = tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True, add_special_tokens=True)
offsets = encoded.pop("offset_mapping")[0]
input_ids = encoded["input_ids"]
# Tokenize input for BERT
# tokens = tokenizer(sentence, return_tensors='pt', return_offsets_mapping=True, add_special_tokens=True)
# input_ids = tokens["input_ids"]
# offsets = tokens["offset_mapping"][0]


# Get attentions
with torch.no_grad():
    outputs = model(**encoded)
    attentions = outputs.attentions  # List of [batch, heads, tokens, tokens]


# Map spaCy tokens to BERT tokens
def find_token_index(word, offsets, text):
    for i, (start, end) in enumerate(offsets):
        if text[start:end] == word:
            return i
    return None


for adj, noun in adj_noun_pairs:
    adj_idx = find_token_index(adj, offsets, sentence)
    noun_idx = find_token_index(noun, offsets, sentence)
   
    if adj_idx and noun_idx:
        # Example: attention from noun to adjective in last layer, head 0
        attn_score = attentions[-1][0, 0, noun_idx, adj_idx].item()
        print(f"Attention from noun '{noun}' to adj '{adj}' (last layer, head 0): {attn_score:.4f}")