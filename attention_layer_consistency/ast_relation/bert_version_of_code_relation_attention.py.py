import ast
import sys
import csv
import time 
import json 
import re 
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np 
from typing import List, Dict, Any, Tuple, Generator
from datasets import load_dataset

# --- 1. AST Analysis and Property Extraction ---

class ASTNode:
    def __init__(self, node_type, start_byte, end_byte, lineno, col_offset,
                 end_lineno=None, end_col_offset=None, parent=None):
        self.node_type = node_type
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.lineno = lineno
        self.col_offset = col_offset
        self.end_lineno = end_lineno
        self.end_col_offset = end_col_offset
        self.parent: ASTNode | None = parent # Link to parent ASTNode object
        self.children: list[ASTNode] = [] # List of child ASTNode objects
        self.original_node = None # Optional: Store the original ast.AST node

    def __repr__(self):
        return (f"ASTNode(type='{self.node_type}', span=({self.start_byte}, {self.end_byte}), "
                f"loc=({self.lineno},{self.col_offset})-({self.end_lineno},{self.end_col_offset}), "
                f"parent='{self.parent.node_type if self.parent else None}')")

class PythonASTAnalyzer(ast.NodeVisitor):
    def __init__(self, code_snippet: str):
        # Store the code as a string and bytes for different uses
        self.code_snippet = code_snippet
        self.code_bytes = code_snippet.encode('utf-8')
        self.ast_nodes: list[ASTNode] = []
        self._node_stack: list[ASTNode] = [] # Stack to keep track of parent nodes

    def build_simplified_ast(self) -> list[ASTNode]:
        """
        Parses the stored Python code snippet and builds a list of ASTNode objects
        with parent-child relationships.

        Returns:
            A list of ASTNode objects representing the flattened AST structure.
        Raises:
            SyntaxError: If the code snippet is not valid Python syntax.
        """
        self.ast_nodes = []
        self._node_stack = []

        try:
            tree = ast.parse(self.code_snippet)
            # The root 'Module' node doesn't have location info in older Python,
            # but we can create a root node representing the whole span.
            root_node = ASTNode(
                node_type=type(tree).__name__,
                start_byte=0,
                end_byte=len(self.code_bytes),
                lineno=1, col_offset=0,
                end_lineno=self.code_bytes.count(b'\n') + 1, # Approximate end line
                end_col_offset=len(self.code_bytes) - self.code_bytes.rfind(b'\n') - 1 if b'\n' in self.code_bytes else len(self.code_bytes) # Approximate end col
            )
            root_node.original_node = tree
            self.ast_nodes.append(root_node)
            self._node_stack.append(root_node)
            for body_item in tree.body:
                 self.generic_visit(body_item)

            self._node_stack.pop()

        except SyntaxError as e:
            raise e

        return self.ast_nodes

    def generic_visit(self, node: ast.AST):
        if not hasattr(node, 'lineno'):
             if not self._node_stack:
                 placeholder_parent = ASTNode("Placeholder", -1, -1, -1, -1)
                 self._node_stack.append(placeholder_parent)
                 super().generic_visit(node)
                 self._node_stack.pop()
             else:
                 super().generic_visit(node)
             return

        try:
            node_source_segment = ast.get_source_segment(self.code_snippet, node, padded=False)
            if node_source_segment is None:
                 start_byte = -1 
                 end_byte = -1   
            else:
                 start_byte = self.code_bytes.find(node_source_segment.encode('utf-8'), node.col_offset)
                 end_byte = start_byte + len(node_source_segment.encode('utf-8'))

        except Exception as e:
             start_byte = -1 
             end_byte = -1   


        end_lineno = getattr(node, 'end_lineno', node.lineno)
        end_col_offset = getattr(node, 'end_col_offset', node.col_offset + (end_byte - start_byte) if start_byte != -1 and end_byte != -1 else node.col_offset)


        parent_node = self._node_stack[-1] if self._node_stack else None

        current_ast_node = ASTNode(
            node_type=type(node).__name__,
            start_byte=start_byte,
            end_byte=end_byte,
            lineno=node.lineno,
            col_offset=node.col_offset,
            end_lineno=end_lineno,
            end_col_offset=end_col_offset,
            parent=parent_node
        )
        current_ast_node.original_node = node

        self.ast_nodes.append(current_ast_node)
        if parent_node:
            parent_node.children.append(current_ast_node)

        self._node_stack.append(current_ast_node)
        super().generic_visit(node)
        self._node_stack.pop()


def map_tokens_to_ast_nodes(code_bytes: bytes, tokens_with_spans: list[tuple[str, tuple[int, int]]],
                            ast_nodes: list[ASTNode]) -> list[list[ASTNode]]:

    token_node_mapping: list[list[ASTNode]] = [[] for _ in range(len(tokens_with_spans))]

    valid_ast_nodes = sorted([n for n in ast_nodes if n.start_byte != -1 and n.end_byte != -1],
                             key=lambda n: (n.start_byte, n.end_byte - n.start_byte))


    for token_idx, token_info in enumerate(tokens_with_spans):
        token_text, (token_start, token_end) = token_info 

        if token_start == -1 or token_end == -1:
             continue

        containing_nodes = []
        for node in valid_ast_nodes:
            if node.start_byte <= token_start and token_end <= node.end_byte:
                containing_nodes.append(node)

        if containing_nodes:
             token_node_mapping[token_idx] = containing_nodes 

    return token_node_mapping

def get_deepest_node_for_token(token_node_mapping: list[list[ASTNode]], token_idx: int) -> ASTNode | None:
    if token_idx < len(token_node_mapping) and token_node_mapping[token_idx]:
        return token_node_mapping[token_idx][-1]
    return None


def generate_parent_child_matrix(num_tokens: int, token_node_mapping: list[list[ASTNode]],
                                 tokens_with_spans: list[tuple[str, tuple[int, int]]]) -> torch.Tensor:

    symbolic_matrix = torch.zeros((num_tokens, num_tokens), dtype=torch.float)

    for child_token_idx in range(num_tokens):
        child_node = get_deepest_node_for_token(token_node_mapping, child_token_idx)

        if child_node and child_node.parent and child_node.parent.start_byte != -1: # Ensure parent node has valid span
            parent_node = child_node.parent
            parent_token_indices = [
                i for i in range(num_tokens)
                if tokens_with_spans[i][1][0] != -1 and tokens_with_spans[i][1][1] != -1 and # Ensure token has valid span
                   parent_node.start_byte <= tokens_with_spans[i][1][0] and tokens_with_spans[i][1][1] <= parent_node.end_byte
            ]

            for parent_token_idx in parent_token_indices:
                 symbolic_matrix[child_token_idx][parent_token_idx] = 1.0

    return symbolic_matrix


def get_model_attention(model, tokenizer, code_snippet: str) -> Tuple[List[str], List[Tuple[int, int]], torch.Tensor]:
    inputs = tokenizer(code_snippet, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=512)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_spans = [(o[0] if o is not None else -1, o[1] if o is not None else -1) for o in inputs["offset_mapping"][0]]

    if 'offset_mapping' in inputs:
        del inputs['offset_mapping']

    # Get attention weights
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    last_layer_attention = outputs.attentions[-1]

    return tokens, token_spans, last_layer_attention[0] 


def extract_attention_by_mask(attention_matrix: torch.Tensor, mask_matrix: torch.Tensor) -> list[float]:
    if attention_matrix.shape != mask_matrix.shape:
        raise ValueError("Attention matrix and mask matrix must have the same shape.")

    # Use the mask matrix as a mask to select the attention values
    relevant_attention_values = attention_matrix[mask_matrix == 1].tolist()

    return relevant_attention_values


def analyze_single_snippet(model_name: str, model, tokenizer, code_snippet: str, ast_nodes: list[ASTNode], code_bytes: bytes) -> Dict[str, float] | None:
    try:
        tokens, token_spans, last_layer_attention = get_model_attention(model, tokenizer, code_snippet)
        num_tokens = len(tokens)

        tokens_with_spans = list(zip(tokens, token_spans))

        token_node_mapping = map_tokens_to_ast_nodes(code_bytes, tokens_with_spans, ast_nodes)

        parent_child_matrix = generate_parent_child_matrix(num_tokens, token_node_mapping, tokens_with_spans)
        num_related_pairs = int(parent_child_matrix.sum())

        not_related_matrix = (parent_child_matrix == 0).float()
        num_not_related_pairs = int(not_related_matrix.sum())

        if num_related_pairs == 0 or num_not_related_pairs == 0:
             return None


        num_heads = last_layer_attention.shape[0]
        all_related_attention = []
        for head_idx in range(num_heads):
            attention_for_head = last_layer_attention[head_idx, :, :] # shape (seq_len, seq_len)
            related_attention_values = extract_attention_by_mask(attention_for_head, parent_child_matrix)
            all_related_attention.extend(related_attention_values)

        all_not_related_attention = []
        for head_idx in range(num_heads):
            attention_for_head = last_layer_attention[head_idx, :, :] # shape (seq_len, seq_len)
            not_related_attention_values = extract_attention_by_mask(attention_for_head, not_related_matrix)
            all_not_related_attention.extend(not_related_attention_values)

        mean_related_attention = np.mean(all_related_attention) if all_related_attention else 0
        mean_not_related_attention = np.mean(all_not_related_attention) if all_not_related_attention else 0

        return {
            f"{model_name} related": mean_related_attention,
            f"{model_name} unrelated": mean_not_related_attention,
            f"{model_name} related - {model_name} unrelated": mean_related_attention - mean_not_related_attention,
        }


    except Exception as e:
        return None


def get_code_snippets_from_dataset(dataset_name: str, split: str, language: str, num_snippets: int | None = None, print_invalid_snippets: bool = False) -> Generator[str, None, None]:
    print(f"Loading dataset: {dataset_name}, split: {split}")
    try:
        dataset = load_dataset(dataset_name, split=split)
        print(f"Dataset split loaded. Number of items: {len(dataset)}") # Print dataset size

        count = 0 
        code_block_regex = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)


        for item_index, item in enumerate(dataset): 
            if num_snippets is not None and count >= num_snippets:
                break

            code_snippet_with_text = item.get('response') 
            snippet_language = item.get('programming_language') 

            if snippet_language is not None and snippet_language.lower() == language.lower():
                if code_snippet_with_text and isinstance(code_snippet_with_text, str):
                    extracted_code = ""
                    match = code_block_regex.search(code_snippet_with_text)

                    if match:
                         extracted_code = match.group(1).strip()

                    if extracted_code:
                        try:
                            ast.parse(extracted_code)
                            yield extracted_code
                            count += 1
                        except SyntaxError as e:
                            if print_invalid_snippets: 
                                print(f"  [Generator] Skipping item {item_index} due to AST SyntaxError in extracted code: {e}")
                            continue 
                        except Exception as e:
                            if print_invalid_snippets: 
                                print(f"  [Generator] Skipping item {item_index} due to unexpected AST error in extracted code during pre-check: {e}")
                            continue 
                    else:
                         if print_invalid_snippets: 
                              print(f"  [Generator] Skipping item {item_index}: No valid code block found in response.")
                         continue 
            else:
                 if print_invalid_snippets: 
                      print(f"  [Generator] Skipping item {item_index}: Language mismatch ('{snippet_language}' != '{language}').")
                 continue 



    except Exception as e:
        print(f"Error loading or reading dataset {dataset_name}: {e}")
        sys.exit(1)



if __name__ == "__main__":
    NUM_SNIPPETS_TO_ANALYZE = 10000 
    DATASET_NAME = "nampdn-ai/tiny-codes" 
    DATASET_SPLIT = "train" 
    DATASET_LANGUAGE = "python" 
    OUTPUT_CSV_FILE = "attention_analysis_results_tinycodes_batch.csv" 
    CODEBERT_MODEL_NAME = "microsoft/CodeBERT-base"
    BERT_MODEL_NAME = "bert-base-uncased" 
    PRINT_INVALID_SNIPPETS = False


    print("Loading CodeBERT and BERT models and tokenizers...")
    try:
        codebert_tokenizer = AutoTokenizer.from_pretrained(CODEBERT_MODEL_NAME)
        codebert_model = AutoModel.from_pretrained(CODEBERT_MODEL_NAME)
        bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
        bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
        print("Models and tokenizers loaded.")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure you have an internet connection and the model names are correct.")
        sys.exit(1)


    all_results: List[Dict[str, Any]] = []
    start_time = time.time()
    processed_snippets_count = 0 # Keep track of total snippets attempted from the dataset iterator
    successfully_analyzed_count = 0 # Keep track of how many snippets made it to all_results

    
    csv_headers = [
        "CodeBERT related", "CodeBERT unrelated", "CodeBERT related - CodeBERT unrelated",
        "BERT related", "BERT unrelated", "BERT related - BERT unrelated",
        "CodeBERT related - BERT related"
    ]

    print(f"\nReading snippets from {DATASET_NAME} ({DATASET_SPLIT}, {DATASET_LANGUAGE}) and starting analysis...")


    for i, code_snippet in enumerate(get_code_snippets_from_dataset(DATASET_NAME, DATASET_SPLIT, DATASET_LANGUAGE, NUM_SNIPPETS_TO_ANALYZE, print_invalid_snippets=PRINT_INVALID_SNIPPETS)):
        processed_snippets_count += 1 # Increment counter for every snippet attempted

        ast_analyzer = PythonASTAnalyzer(code_snippet)
        try:
            ast_nodes = ast_analyzer.build_simplified_ast()
            code_bytes = ast_analyzer.code_bytes # Get code bytes from analyzer instance
        except Exception as e:

             if PRINT_INVALID_SNIPPETS: # Only print if the flag is True
                 print(f"Skipping snippet {processed_snippets_count} due to unexpected AST error AFTER generator filter: {e}")
             continue


        codebert_analysis = analyze_single_snippet("CodeBERT", codebert_model, codebert_tokenizer, code_snippet, ast_nodes, code_bytes)

        bert_analysis = analyze_single_snippet("BERT", bert_model, bert_tokenizer, code_snippet, ast_nodes, code_bytes)

        if codebert_analysis is not None and bert_analysis is not None:
            combined_result = {**codebert_analysis, **bert_analysis}

            codebert_related_key = "CodeBERT related" 
            bert_related_key = "BERT related" 
            if codebert_related_key in combined_result and bert_related_key in combined_result:
                 combined_result["CodeBERT related - BERT related"] = combined_result[codebert_related_key] - combined_result[bert_related_key]
                 all_results.append(combined_result)
                 successfully_analyzed_count += 1 



        if successfully_analyzed_count > 0 and successfully_analyzed_count % 100 == 0:
             elapsed_time = time.time() - start_time
             snippets_per_second = processed_snippets_count / elapsed_time if elapsed_time > 0 else 0
             print(f"Processed {processed_snippets_count} snippets (Attempted) | Successfully analyzed: {successfully_analyzed_count} | Elapsed: {elapsed_time:.2f}s | Speed: {snippets_per_second:.2f} snippets/s")


    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print(f"\nAnalysis complete for {successfully_analyzed_count} successfully analyzed snippets out of {processed_snippets_count} attempted.")
    print(f"Total time elapsed: {total_elapsed_time:.2f} seconds.")

    if all_results:
        print(f"\nWriting results to {OUTPUT_CSV_FILE}...")
        try:
            with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                writer.writeheader()
                writer.writerows(all_results)
            print("CSV file written successfully.")
        except IOError as e:
            print(f"Error writing CSV file: {e}")
    else:
        print("\nNo successful analysis results to write to CSV.")


    if all_results:
        print("\n--- Average Values Across Analyzed Snippets ---")

        try:
            import pandas as pd
            df = pd.DataFrame(all_results)
            df = df[csv_headers]
            average_values = df.mean().to_dict()
            for header in csv_headers:
                print(f"Average '{header}': {average_values.get(header, 0):.6f}")
        except ImportError:
            print("Pandas not found. Calculating averages manually (requires numpy).")
            if np is not None:
                filtered_results = [{header: d.get(header, np.nan) for header in csv_headers} for d in all_results]
                data_for_avg = np.array([[d.get(header, np.nan) for header in csv_headers] for d in filtered_results], dtype=float)

                mean_values = np.nanmean(data_for_avg, axis=0)

                for i, header in enumerate(csv_headers):
                     print(f"Average '{header}': {mean_values[i]:.6f}")
            else:
                 print("Numpy not found. Cannot calculate averages.")

    else:
        print("\nNo successful analysis results to calculate averages.")