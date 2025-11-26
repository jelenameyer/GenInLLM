#!/usr/bin/env python3
"""
Sequential Prompting Task Module - Prompts LLM one item at a time with context.
Extracts both actual model answers and log probabilities for all alternatives.
"""
# ------------- load packages and files --------------

import json
import torch
import torch.nn.functional as F
import pandas as pd
import re
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from base_task import encode_without_chat_template
from utilsBehaviouralTasks import detect_chat_tokens
from typing import Literal
import outlines
import os

DATA_FILE = "survey_data/subset_direct_LLM_prompts.jsonl"
TASK_NAME = "Sequential_LLM_Prompting"
MODEL_NAME =  "swiss-ai/Apertus-70B-Instruct-2509"

# ------------- Helper Functions -------------------


def extract_answer_range_from_experiment(experiment: str, flipped: Union[str, bool]) -> Tuple:
    """
    Extract answer range from experiment type.
    Returns tuple of valid answers (can be ints or strings).
    """
    # Convert flipped to boolean if it's a string
    # is_flipped = flipped in [True, "true", "yes"]
    
    if experiment == "BARRAT scale":
        return tuple(range(1, 5))  # 1,2,3,4
    elif experiment == "DOSPERT scale":
        return tuple(range(1, 6))  # 1,2,3,4,5
    elif experiment == "SOEP scale":
        return tuple(range(1, 12))  # 1-11
    elif experiment == "SSSV scale":
        return (1, 2)
    elif experiment == "Decisions From Description":
        return (1, 2)
    else:
        logging.warning("no answer range extracted, unknown experiment!")
        return tuple()


def parse_items_from_text(text: str, experiment: str, flipped: Union[str, bool]) -> List[Dict[str, Any]]:
    """
    Parse individual items from the questionnaire text.
    Returns list of dicts with: {item_num, question, answer_range}
    """
    items = []
    answer_range = extract_answer_range_from_experiment(experiment, flipped)
    
    # Different patterns for different experiment types
    if experiment == "Decisions From Description":
        problem_pattern = r"Problem (\d+):(.*?)You choose Box <<([XRTA])>>"
        matches = re.finditer(problem_pattern, text, re.DOTALL)
        
        for match in matches:
            item_num, question_text, human_answer = match.groups()
            items.append({
                'item_num': int(item_num),
                'question': question_text.strip(),
                'answer_range': answer_range
            })
    
    elif experiment == "SSSV scale":
        # SSSV has paired statements - need special handling
        # Pattern: "1 = 'statement', 2 = 'statement' <<answer>>"
        statement_pattern = r"(\d+)\s*=\s*'([^']+)',\s*(\d+)\s*=\s*'([^']+)'\s*<<(\d+)>>"
        matches = re.finditer(statement_pattern, text)
        
        item_counter = 1
        for match in matches:
            num1, stmt1, num2, stmt2, answer = match.groups()
            question_text = f"{num1} = '{stmt1}', {num2} = '{stmt2}'"
            items.append({
                'item_num': item_counter,
                'question': question_text,
                'answer_range': answer_range
            })
            item_counter += 1
    
    else:
        # Standard pattern: "number. question text <<answer>>"
        pattern = r"(\d+)\.\s*(.+?)\s*<<(\d+)>>"
        matches = re.finditer(pattern, text)
        
        for match in matches:
            item_num, question_text, human_answer = match.groups()
            items.append({
                'item_num': int(item_num),
                'question': question_text.strip(),
                'answer_range': answer_range,
                'human_answer': human_answer
            })
    
    return items


def extract_instruction(text: str, experiment_type: str) -> str:
    """Extract instruction from questionnaire text."""
    if experiment_type == "survey":
        instruction_match = re.search(
            r"Instructions?:?\s*\n(.*?(?:\nStatements?:|$))",
            text,
            re.DOTALL
        )
        instructions = instruction_match.group(1).strip() if instruction_match else ""
    elif experiment_type == "behaviour":
        instruction_match = re.search(
            r"(.*?(?:\n\nProblem 1:|$))",
            text,
            re.DOTALL
        )
        instructions = instruction_match.group(1).strip() if instruction_match else ""
    else:
        logging.warning("no INSTRUCTION extracted, unknown experiment type!")
        return ""
    return instructions




def build_prompt_with_context(
    instruction: str,
    current_item: Dict[str, Any],
    previous_items: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    entry_idx: int,
    experiment: str
) -> str:
    """
    Build prompt with instruction, previous Q&As, and current question.
    """
    USER_TOK, ASSIST_TOK = detect_chat_tokens(tokenizer)
    
    prompt_parts = []
    
    # Add instruction 
    if instruction: 
        prompt_parts.append(f"{instruction}")
    
    # Add previous items with their model answers for context
    for i, prev_item in enumerate(previous_items):
        # The very first ever question (first in the whole sequence) has no USER_TOK
        if i == 0:
            prompt_parts.append(
                f"{prev_item['item_num']}. {prev_item['question']} <<"
            )
        else:
            # All other previous questions get USER_TOK
            prompt_parts.append(
                f"{USER_TOK} {prev_item['item_num']}. {prev_item['question']} <<"
            )

        prompt_parts.append(f"{ASSIST_TOK} Response: {prev_item['model_answer']}")

    # Add current item (the one we’re about to generate an answer for)
    # If this is the first item overall (no previous_items), do NOT add USER_TOK
    if len(previous_items) == 0:
        prompt_parts.append(
            f"{current_item['item_num']}. {current_item['question']} <<"
        )
    else:
        prompt_parts.append(
            f"{USER_TOK} {current_item['item_num']}. {current_item['question']} <<"
        )
    # Add assistant token with "Response: " prefix to start generation
    # prompt_parts.append(f"{ASSIST_TOK} Response:") # change here? 
    return "\n".join(prompt_parts)


def sequence_logprob(model, tokenizer, prompt_input_ids: torch.LongTensor, answer_str: str, device=None):
    """
    Compute exact log-probability of `answer_str` given the prompt_input_ids for causal LMs.
    Returns total log probability (sum over tokens).
    """
    device = device or next(model.parameters()).device
    model.eval()

    # tokenize answer (no special tokens)
    ans_ids = tokenizer(answer_str, add_special_tokens=False).input_ids
    if len(ans_ids) == 0:
        return float('-inf')

    total_lp = 0.0
    prefix = prompt_input_ids.clone().to(device)

    # iterate token by token
    for tok in ans_ids:
        outputs = model(prefix.unsqueeze(0))
        logits = outputs.logits[0, -1, :]
        logprobs = F.log_softmax(logits, dim=-1)
        total_lp += logprobs[tok].item()
        prefix = torch.cat([prefix, torch.tensor([tok], device=device)], dim=0)

    return total_lp





def get_model_answer_and_logprobs_chat(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    answer_range: Tuple,
    temperature: float = 0.0,
    max_new_tokens: int = 10
) -> Tuple[str, Dict[str, float]]:
    """
    Generate model's full response using Outlines for structured output,
    and compute log-probabilities for all valid answer options.
    """
    model.eval()
    outlines_model = outlines.from_transformers(
        AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto"),
        AutoTokenizer.from_pretrained(MODEL_NAME)
    )

    # ---------- Build chat message ----------
    messages = [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": "Response: "},
    ]

    # ----- Create the chat string depending on if model has chat template or not -----
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        chat_str = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        chat_str = prompt
    # print("=== Model sees this exact text ===")
    # print(chat_str)
    # print("===================================")

    # ====== Use Outlines for structured decoding ======

    # Wrap the model if not already wrapped
    #outlines_model = outlines.from_transformers(model, tokenizer)

    # Define the allowed answer values as Literal types
    allowed_answers = Literal[tuple(str(a) for a in answer_range)]

    # Generate the structured answer (guaranteed to be one of the allowed options)
    extracted_answer = outlines_model(
        chat_str,
        allowed_answers,
        max_new_tokens=max_new_tokens
    )
    extracted_answer = str(extracted_answer).strip()
    # print(f"Structured answer (Outlines): {extracted_answer}")

    # ----- Compute log probabilities for all alternatives -----
    inputs = tokenizer(chat_str, return_tensors="pt").to(model.device)
    logprobs_dict = {}
    input_ids = inputs["input_ids"][0]

    for ans in answer_range:
        ans_str = str(ans)
        lp = sequence_logprob(model, tokenizer, input_ids, ans_str, model.device)
        logprobs_dict[f"logprob_{ans_str}"] = lp

    return extracted_answer, logprobs_dict




def process_questionnaire_sequential(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_key: str,
    entry_idx: int,
    experiment: str = "",
    exp_type: str = "",
    flipped: Union[str, bool] = False
) -> List[Dict[str, Any]]:
    """
    Process questionnaire by prompting model item-by-item sequentially.
    
    Returns:
        List of result dicts with model answers and logprobs.
    """
    # Extract instruction and items
    instruction = extract_instruction(text, exp_type)
    #print(f"INSTRUCTIONS: {instruction}")
    items = parse_items_from_text(text, experiment, flipped)
    
    if not items:
        logging.warning(f"No items parsed for {experiment}")
        return []
    
    results = []
    previous_items = []
    
    for item in items:
        # Build prompt with context
        prompt = build_prompt_with_context(
            instruction,
            item,
            previous_items,
            tokenizer,
            entry_idx,
            experiment
        )
        #print(prompt, "\n", "\n","\n")
        # Get model answer and logprobs
        model_answer, logprobs = get_model_answer_and_logprobs_chat(
            prompt,
            model,
            tokenizer,
            item['answer_range'],
            temperature=0.0  # Greedy decoding
        )
        
        # Store model answer for next iteration's context
        item['model_answer'] = model_answer
        previous_items.append(item)
        
        # Create result row
        result = {
            'model': model_key,
            'experiment': experiment,
            'type': exp_type,
            'flipped': str(flipped),
            'item': item['item_num'],
            'question': item['question'],
            'model_answer': model_answer,
            **logprobs  # Add all logprobs as separate columns
        }
        
        results.append(result)
    
    return results


# ------------- Main Task Runner (called by model_manager.py) -----
def run_task(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_key: str,
    test_mode: bool = False,
    data_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Main task runner function - called by the model manager.
    
    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        model_key: String identifier for the model
        test_mode: Whether to run in test mode (fewer entries, more logging)
        data_file: Optional override for data file path
        
    Returns:
        pandas.DataFrame: Results with model answers and logprobs
    """
    data_path = data_file or DATA_FILE
    logging.info(f"Starting {TASK_NAME} task for model: {model_key}")
    
    all_results = []
    
    try:
        # Load data
        with open(data_path) as f:
            entries = [json.loads(line) for line in f]
        
        # Limit entries in test mode
        if test_mode:
            entries = entries[:2]
            logging.info(f"Test mode: processing only first 2 entries")
        
        # Process each questionnaire
        for entry_idx, entry in enumerate(entries):
            logging.info(
                f"Processing {TASK_NAME} entry {entry_idx + 1}/{len(entries)}: "
                f"{entry.get('experiment', 'Unknown')}"
            )
            
            try:
                results = process_questionnaire_sequential(
                    text=entry['text'],
                    model=model,
                    tokenizer=tokenizer,
                    model_key=model_key,
                    entry_idx=entry_idx,
                    experiment=entry.get('experiment', ''),
                    exp_type=entry.get('type', ''),
                    flipped=entry.get('flipped', False)
                )
                
                all_results.extend(results)
                
                if test_mode and results:
                    logging.info(f"Sample result: {results[0]}")
                
            except Exception as e:
                logging.error(
                    f"Error processing {TASK_NAME} entry {entry_idx} "
                    f"({entry.get('experiment', 'Unknown')}): {e}",
                    exc_info=True
                )
                continue
        
        # Create DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            logging.info(
                f"{TASK_NAME} task completed. "
                f"Generated {len(all_results)} rows of results."
            )
            return df
        else:
            logging.warning(
                f"No results generated for {TASK_NAME} task on {model_key}"
            )
            return pd.DataFrame()
            
    except Exception as e:
        logging.error(
            f"Error in {TASK_NAME} task for model {model_key}: {e}",
            exc_info=True
        )
        return pd.DataFrame()

# ------------- Main script entry point -------------
if __name__ == "__main__":
    import argparse
    import sys

    # -------- Argument parser --------
    parser = argparse.ArgumentParser(
        description="Run Sequential Prompting Task directly without wrapper."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=MODEL_NAME,
        help="Hugging Face model name or local path (e.g., 'gpt2' or './my_model')."
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=DATA_FILE,
        help=f"Path to JSONL data file (default: {DATA_FILE})."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="sequential_prompting_results.csv",
        help="Where to save results (CSV)."
    )
    
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run only first few entries for testing."
    )

    args = parser.parse_args()

    # -------- Logging setup --------
    log_file = "sequential_prompting.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),  # save logs here
            logging.StreamHandler(sys.stdout)         # also print to console
        ]
    )

    # -------- Load model and tokenizer --------
    logging.info(f"Loading model and tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")

    # Set global model name if used elsewhere
    MODEL_NAME = args.model_name

    # -------- Run the task --------
    df = run_task(
        model=model,
        tokenizer=tokenizer,
        model_key=args.model_name,
        test_mode=args.test_mode,
        data_file=args.data_file
    )

    # -------- Save results --------
    if not df.empty:
        df.to_csv(args.output_file, index=False)
        logging.info(f"Results saved to: {args.output_file}")
    else:
        logging.warning("No results generated. Nothing was saved.")

