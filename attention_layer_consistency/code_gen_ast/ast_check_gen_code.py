import ast
import sys
import time
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from typing import List

def check_ast(code: str) -> int:
    if not code.strip():
        return 0
    try:
        ast.parse(code)
        return 1
    except (SyntaxError, ValueError):
        return 0
    except Exception as e:
        print(f"Unexpected AST parsing error: {e}")
        return 0

if __name__ == "__main__":
    PROMPTS_FILE = "prompts.txt"
    OUTPUT_LOG_FILE = "output.log"
    CODE_GENERATION_MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"

    TRIPLE_BACKTICK_REGEX = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)
    LATEX_CODE_MARKER_REGEX = re.compile(r"\\begin\{code\}\n(.*?)\n\\end\{code\}", re.DOTALL)

    OUTPUT_SEPARATOR = "\n" + "="*80 + "\n"

    code_generation_model = None
    code_generation_tokenizer = None

    print(f"Loading code generation model: {CODE_GENERATION_MODEL_ID}...")
    try:
        code_generation_tokenizer = AutoTokenizer.from_pretrained(CODE_GENERATION_MODEL_ID)
        code_generation_model = AutoModelForCausalLM.from_pretrained(CODE_GENERATION_MODEL_ID)
        print("Code generation model loaded.")
    except Exception as e:
        print(f"Error loading code generation model {CODE_GENERATION_MODEL_ID}: {e}")
        print("Please ensure you have accepted the model's terms on Hugging Face Hub and are logged in (`huggingface-cli login`).")
        print("Exiting.")
        sys.exit(1)

    prompts: List[str] = []
    try:
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            print(f"No prompts found in {PROMPTS_FILE}. Exiting.")
            sys.exit(1)
        print(f"Read {len(prompts)} prompts from {PROMPTS_FILE}.")
    except FileNotFoundError:
        print(f"Error: Prompts file '{PROMPTS_FILE}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading prompts file {PROMPTS_FILE}: {e}")
        sys.exit(1)

    total_prompts = len(prompts)
    ast_check_results: List[int] = []
    total_processed_count = 0
    start_time = time.time()

    print(f"\nStarting code generation and AST check for {total_prompts} prompts (once each)...")
    print(f"Writing extracted code output to {OUTPUT_LOG_FILE}")

    with open(OUTPUT_LOG_FILE, 'a', encoding='utf-8') as log_file:
        for i, prompt in enumerate(prompts):
            total_processed_count += 1
            print(f"\nProcessing Prompt {i + 1}/{total_prompts}...")
            print(f"Prompt: {prompt}")

            generation_start_time = time.time()
            full_generated_text = ""

            try:
                inputs = code_generation_tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to("cpu")
                code_generation_model.to("cpu")
                code_generation_model.eval()
                with torch.no_grad():
                    generated_ids = code_generation_model.generate(
                        **inputs,
                        max_new_tokens=256,
                        num_return_sequences=1,
                        pad_token_id=code_generation_tokenizer.eos_token_id,
                    )
                full_generated_text = code_generation_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                generation_end_time = time.time()
                generation_time = generation_end_time - generation_start_time
                print(f"  Code generation time: {generation_time:.2f} seconds")

                raw_generated_code = ""
                extraction_method = "None"

                code_block_match = re.search(TRIPLE_BACKTICK_REGEX, full_generated_text)
                if code_block_match:
                    raw_generated_code = code_block_match.group(1).strip()
                    extraction_method = "Triple Backticks"
                    print("  Extracted code block using Triple Backticks (```).")
                else:
                    code_block_match = re.search(LATEX_CODE_MARKER_REGEX, full_generated_text)
                    if code_block_match:
                        raw_generated_code = code_block_match.group(1).strip()
                        extraction_method = "\\begin{code}...\\end{code}"
                        print("  Extracted code block using \\begin{code}...\\end{code} markers.")
                    else:
                        raw_generated_code = full_generated_text.strip()
                        extraction_method = "Full Text (No Markers Found)"
                        print("  Code markers not found. Using entire generated text for AST check.")

                if extraction_method in ["Triple Backticks", "\\begin{code}...\\end{code}"]:
                    print(f"  Writing extracted code ({extraction_method}) for prompt {i+1} to {OUTPUT_LOG_FILE}...")
                    log_file.write(f"--- Prompt {i+1}/{total_prompts} ---\n")
                    log_file.write(f"Prompt: {prompt}\n")
                    log_file.write(f"Extraction Method: {extraction_method}\n")
                    log_file.write(f"Extracted Code Output:\n")
                    log_file.write(raw_generated_code)
                    log_file.write(OUTPUT_SEPARATOR)
                    log_file.flush()
                else:
                    print(f"  No code block extracted by markers for prompt {i+1}. Not logging code to {OUTPUT_LOG_FILE}.")

                ast_pass_fail = check_ast(raw_generated_code)
                print(f"  Generated Code AST Check: {'Pass' if ast_pass_fail else 'Fail'} ({ast_pass_fail})")
                ast_check_results.append(ast_pass_fail)

            except Exception as e:
                print(f"  An error occurred during code generation or analysis for prompt {i+1}: {e}")
                print(f"  Generated Code AST Check: Fail (0)")
                ast_check_results.append(0)
                error_log_content = f"--- Prompt {i+1}/{total_prompts} ---\nPrompt: {prompt}\nError processing Prompt {i+1}: {e}\n"
                if 'full_generated_text' in locals() and full_generated_text:
                     error_log_content += f"Note: Full generated text was produced but an error occurred during processing.\n"
                log_file.write(error_log_content)
                log_file.write(OUTPUT_SEPARATOR)
                log_file.flush()

    end_time = time.time()
    total_elapsed_time = end_time - start_time
    print(f"\nAnalysis complete for {total_processed_count} prompts.")
    print(f"Total time elapsed: {total_elapsed_time:.2f} seconds.")

    total_runs = len(ast_check_results)
    passed_runs = sum(ast_check_results)
    ast_pass_percentage = (passed_runs / total_runs) * 100 if total_runs > 0 else 0

    print(f"\n--- Final AST Check Statistics ---")
    print(f"Total Prompts Processed: {total_runs}")
    print(f"Generated Codes Passing AST Check: {passed_runs}")
    print(f"Percentage Passing AST Check: {ast_pass_percentage:.2f}%")
