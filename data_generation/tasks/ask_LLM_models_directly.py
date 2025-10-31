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

DATA_FILE = "survey_data/direct_LLM_prompts.jsonl"
TASK_NAME = "Sequential_LLM_Prompting"


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


def extract_answer_from_response(response_text: str, answer_range: Tuple) -> str:
    """
    Extract the actual answer from model's generated response.
    Handles various formats like "Response: 3", "3", "The answer is 3", etc.
    """
    response_text = response_text.strip()
    
    # Try to find any valid answer in the response
    for ans in answer_range:
        ans_str = str(ans)
        # Check if answer appears as standalone or with common prefixes
        patterns = [
            rf"^{re.escape(ans_str)}$",  # Just the answer
            rf"^Response:\s*{re.escape(ans_str)}",  # "Response: X"
            rf"^{re.escape(ans_str)}\b",  # Answer at start with word boundary
            rf"\b{re.escape(ans_str)}$",  # Answer at end with word boundary
        ]
        for pattern in patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                return ans_str
    
    # If no match, return the empty token/word as fallback
    first_token = response_text.split()[0] if response_text else ""
    logging.warning(f"Could not extract valid answer from: '{response_text}', using:''")
    return ""


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
        prompt_parts.append(f"{USER_TOK} {instruction}")
    
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

# def get_token_logprob(
#     logprobs: torch.Tensor,
#     tokenizer,
#     answer_str: str
# ) -> float:
#     """
#     Compute the log probability for an answer string, handling:
#     - multi-token answers (sum logprobs)
#     - space-preferring tokenizations (" A" vs "A")
#     """
#     # Tokenize both with and without leading space
#     ans_with_space = tokenizer(f" {answer_str}", add_special_tokens=False).input_ids
#     ans_without_space = tokenizer(answer_str, add_special_tokens=False).input_ids

#     # Prefer single-token versions, or whichever is shorter
#     if len(ans_without_space) == 1:
#         ans_tokens = ans_without_space
#     elif len(ans_with_space) == 1:
#         ans_tokens = ans_with_space
#     else:
#         ans_tokens = ans_with_space if len(ans_with_space) <= len(ans_without_space) else ans_without_space

#     if len(ans_tokens) == 0:
#         logging.warning(f"No tokens found for answer: {answer_str}")
#         return float('-inf')

#     # For single token, return its logprob directly
#     if len(ans_tokens) == 1:
#         tok = ans_tokens[0]
#         if tok < logprobs.size(0):
#             return logprobs[tok].item()
#         else:
#             logging.warning(f"Token index {tok} out of range for answer: {answer_str}")
#             return float('-inf')

#     # For multiple tokens, sum their logprobs (approximation)
#     total_lp = 0.0
#     for tok in ans_tokens:
#         if tok < logprobs.size(0):
#             total_lp += logprobs[tok].item()
#         else:
#             logging.warning(f"Token index {tok} out of range for answer: {answer_str}")
#             break

#     # warning for debugging
#     if len(ans_tokens) > 1:
#         decoded = tokenizer.decode(ans_tokens)
#         logging.warning(f"Answer '{answer_str}' tokenized into {len(ans_tokens)} tokens ({decoded}) — summed logprobs.")

#     return total_lp

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
    Generate model's full response using chat template, extract answer,
    and compute log-probabilities for all valid answer options.
    """
    model.eval()

    # ---------- get message prompt -----------
    system_prompt = (
        "You are an AI completing psychological questionnaire items. "
        "Always respond only with the number corresponding to your choice, no explanation."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(prompt)},
         {"role": "assistant", "content": "Response: "}
]

    # ----- Apply chat template -----
    chat_str = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )
    # print("=== Model sees this exact text ===")
    # print(chat_str)
    # print("===================================")
    
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
        chat_template="final",
        reasoning = False,
    
    ).to(model.device)

    # ----- Generate -----
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # ----- Extract generated text -----
    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # ----- Extract model's chosen answer -----
    extracted_answer = extract_answer_from_response(generated_text, answer_range)

    # ----- Compute log probabilities for all alternatives -----
    logprobs_dict = {}
    input_ids = inputs["input_ids"][0]
    for ans in answer_range:
        ans_str = str(ans)
        lp = sequence_logprob(model, tokenizer, input_ids, ans_str, model.device)
        logprobs_dict[f"logprob_{ans_str}"] = lp

    return extracted_answer, logprobs_dict


# def get_model_answer_and_logprobs(
#     prompt: str,
#     model: AutoModelForCausalLM,
#     tokenizer: AutoTokenizer,
#     answer_range: Tuple,
#     temperature: float = 0.0,
#     max_new_tokens: int = 10
# ) -> Tuple[str, Dict[str, float]]:
#     """
#     Generate model's full response, extract answer,
#     and compute log-probabilities for all valid answer options.
#     """
#     device = next(model.parameters()).device
#     model.eval()

#     # ----- Encode prompt -----
#     enc = encode_without_chat_template(tokenizer, prompt)
#     input_ids = enc.input_ids.to(device)
#     attention_mask = torch.ones_like(input_ids, device=device)
#     # ----- Generate model response -----
#     with torch.inference_mode():
#         output = model.generate(
#             input_ids,
#             attention_mask=attention_mask,
#             max_new_tokens=max_new_tokens,
#             do_sample=False,
#             pad_token_id=tokenizer.eos_token_id
#         )

#     # decode generated part (after the prompt)
#     generated_tokens = output[0, input_ids.shape[1]:]
#     generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

#     # extract model's chosen answer
#     extracted_answer = extract_answer_from_response(generated_text, answer_range)

#     # ----- Compute log probabilities for all alternatives -----
#     logprobs_dict = {}
#     for ans in answer_range:
#         ans_str = str(ans)
#         lp = sequence_logprob(model, tokenizer, input_ids[0] if input_ids.ndim > 1 else input_ids, ans_str, device)
#         logprobs_dict[f"logprob_{ans_str}"] = lp

#     return extracted_answer, logprobs_dict


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
