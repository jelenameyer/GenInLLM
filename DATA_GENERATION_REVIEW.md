# Data Generation Code Review

## Date: November 22, 2025

## Purpose
Verify the data generation methodology is sound before addressing analysis issues.

---

## Overview of Data Generation Architecture

### System Components

1. **`Calling_LLM_Models.py`** - Main orchestrator
   - Loads models from HuggingFace (44 different models)
   - Runs tasks on each model
   - Manages memory and GPU resources

2. **`base_task.py`** - Framework for survey tasks
   - Provides logprob extraction functionality
   - Handles chat template formatting
   - Processes questionnaire text with `<< >>` markers

3. **Task Modules** (18 total in `/tasks/`)
   - Survey tasks: AUDIT, BARRAT, CARE, DAST, etc.
   - Behavioral tasks: BART, CCT, DFD, DFE, LOT, MPL

---

## Data Generation Process

### For Survey Tasks (e.g., CARE, AUDIT, BARRAT)

**Input Data Structure** (JSONL format):
```json
{
  "text": "1. Question text <<answer>>\n2. Another question <<answer>>...",
  "participant": "participant_id",
  "experiment": "CARE scale",
  "flipped": "true/false"
}
```

**Processing Steps:**

1. **Load participant data** from JSONL file
   - Each entry contains multiple questions with human answers marked by `<<answer>>`

2. **Format for chat template** (`base_task.py:86-124`):
   ```python
   # Converts:
   "1. Question text <<answer>>"

   # Into:
   "<|user|> 1. Question text <<
   <|assistant|> answer"
   ```

3. **Tokenize without chat template** (`base_task.py:18-27`):
   - Temporarily disables chat template to avoid extra tokens
   - Gets offset mapping for token alignment

4. **Extract log probabilities** (`base_task.py:126-225`):
   - Forward pass through LLM
   - Compute softmax → log probabilities
   - For each `<<answer>>` position:
     - Find token indices overlapping the answer span
     - Extract logprob for EACH candidate answer (e.g., 0-99 for CARE)
     - Store as `log_prob_0`, `log_prob_1`, ..., `log_prob_99`

5. **Save results** (`Calling_LLM_Models.py:486-489`):
   ```
   {ModelName}_{TaskName}_prompting_results.csv
   ```
   Contains columns:
   - `model`, `item`, `participant`, `experiment`
   - `human_number` (actual human answer)
   - `log_prob_0`, `log_prob_1`, ..., `log_prob_N` (LLM probabilities for each option)

### For Behavioral Tasks (e.g., BART, CCT)

**Input Data** (JSONL with trial sequences):
```json
{
  "text": "Instructions... Balloon 1: {pump}{pump}{stop}...",
  "participant": "participant_id"
}
```

**Processing** (see `BART_prompting.py`):

1. **Parse trial structure** (lines 21-71):
   - Extract pump/stop keys (randomized letters)
   - Identify decision points marked by `{key}`
   - Track balloon outcomes (explode vs. collect points)

2. **Build chat sequence** (lines 73-179):
   - Format as conversation with LLM responding at each `{decision}`
   - Single forward pass for all decisions in sequence

3. **Extract decision logprobs** (lines 181-249):
   - Get logprob for pump vs. stop at each decision point
   - Match to assistant token positions
   - Store with trial metadata

4. **Save results**:
   - Columns: `model`, `participant`, `round`, `decision_num`
   - `log_prob_pump`, `log_prob_stop`
   - `human_decision` (what human actually did)

---

## Code Quality Assessment

### ✅ Correct Implementations

1. **Log Probability Extraction**
   - Properly uses `torch.nn.functional.log_softmax` (not raw logits)
   - Correctly indexes into logit tensor: `logprobs[tok_idx][candidate_token]`
   - Handles multi-token numbers with summation (approximation but reasonable)

2. **Chat Template Handling**
   - Auto-detects user/assistant tokens from tokenizer
   - Fallback for models without chat templates
   - Temporarily disables template during encoding to avoid token duplication

3. **Token Alignment**
   - Uses `offset_mapping` to find tokens overlapping answer spans
   - Robust to tokenization differences across models

4. **Memory Management**
   - Uses `torch.inference_mode()` during forward pass
   - Clears GPU cache between tasks/models
   - Mid-run model reload to prevent memory fragmentation

### ⚠️ Potential Issues

#### 1. **Multi-Token Number Handling** (`base_task.py:211-216`)

```python
if len(enc) > 1:
    # Multi-token number: sum logprobs (approximation)
    lp = 0.0
    for j, t in enumerate(enc):
        if tok_idx + j < logprobs.size(0):
            lp += logprobs[tok_idx + j][t].item()
```

**Issue**: Summing log probabilities assumes independence, but sequential tokens are NOT independent.

**Impact**:
- Affects how multi-token numbers (e.g., "10", "99") are scored
- Could bias results if some answer options tokenize into multiple tokens
- Different tokenizers split numbers differently

**Severity**: **MEDIUM** - Systematic but likely small effect

**Recommendation**:
- Check which numbers tokenize into multiple tokens per model
- Consider using only the first token logprob (simpler approximation)
- Document this limitation

#### 2. **Token Position Matching** (`base_task.py:172-186`)

```python
# Find token indices overlapping with this number span
tok_indices = [
    i for i, (lo, hi) in enumerate(offsets)
    if not (hi <= span_lo or lo >= span_hi)
]

if not tok_indices:
    logging.warning(
        f"No token overlap for answer {human_answer} at span {span_lo}-{span_hi}"
    )
    continue  # ← SKIPS THIS ANSWER
```

**Issue**: If offset mapping fails, the answer is silently dropped.

**Impact**:
- Missing data for some items
- Could bias results if failures correlate with certain answers or models

**Severity**: **LOW** - Has warning but data loss is silent

**Recommendation**:
- Track how often this happens per model
- Consider raising error if >X% of answers fail

#### 3. **Chat Token Detection Fallback** (`base_task.py:54-73`)

```python
if entry_idx == 0:
    logging.warning('Did not find specific assistant and user tokens, probably no chat models, using none!!')
return "", "<<"  # ← Fallback for non-chat models
```

**Issue**: For models without chat templates, uses empty string for user token and `<<` for assistant.

**Impact**:
- May not work properly with all non-instruct models
- Different formatting could affect logprob extraction quality

**Severity**: **LOW-MEDIUM** - Only affects ~5-10 models (non-instruct variants)

**Recommendation**:
- Verify which models use the fallback
- Check if results differ systematically for those models

#### 4. **Participant Data Dependency** (THE BIG ONE)

**Location**: Throughout the pipeline

**Issue**: LLM log probabilities are extracted **at positions determined by human answers**.

From `base_task.py:250-261`:
```python
for i, s in enumerate(spans, 1):
    s["model"] = model_key
    s["item"] = i
    s["participant"] = entry.get("participant", "")  # ← Human participant ID
    s["human_number"] = ...  # ← Human answer
    rows.append(s)
```

**Critical observation**: The data generation creates one row per `<<answer>>` position in the human data. This means:
1. LLM sees the FULL sequence with human answers filled in
2. LLM probabilities are conditioned on seeing previous human answers
3. Each "model evaluation" is actually evaluating on DIFFERENT human participant sequences

**Example**:
```
Participant 1: Q1:<<3>> Q2:<<5>> Q3:<<2>>
Participant 2: Q1:<<1>> Q2:<<4>> Q3:<<5>>
```

When evaluating LLM on Participant 1's data:
- At Q2, LLM has seen "Q1:<<3>>" in context
- LLM's probability for Q2 may be influenced by Q1's answer
- This is the HUMAN'S answer from Q1, not the LLM's own consistency

**Implications**:
1. ✅ **Good**: LLMs are evaluated on realistic human response patterns
2. ⚠️ **Concern**: LLM probabilities are not "pure" - they're conditional on human context
3. ⚠️ **Major Concern**: This explains why analysis weights by human data feel "natural" - because the data generation ALREADY uses human data structure

**Severity**: **CRITICAL for interpretation**

**Recommendation**: This is not necessarily WRONG, but it's a **design choice** that has implications:
- Current approach: "How well does LLM predict each human answer given that human's previous answers?"
- Alternative: "What is LLM's internal consistency across questions?"

The current approach is reasonable for studying "LLM-human alignment" but may not be ideal for studying "LLM personality consistency."

---

## Data Integrity Checks

### Verified Correct:

1. ✅ **Log probabilities are properly normalized**
   - Uses `log_softmax` not raw logits

2. ✅ **All candidate answers are scored**
   - Loops through full ANSWER_RANGE for each position

3. ✅ **Model outputs are deterministic**
   - Uses `model.eval()` and `torch.inference_mode()`
   - No dropout or sampling

4. ✅ **Participant structure is preserved**
   - Each participant's full sequence is processed together
   - Order of questions maintained

### Needs Verification:

1. ⚠️ **Multi-token scoring consistency**
   - Check distribution of multi-token numbers across models
   - Verify impact on results

2. ⚠️ **Missing data frequency**
   - How often does token alignment fail?
   - Is it consistent across models?

3. ⚠️ **Chat template coverage**
   - Which models use fallback tokens?
   - Do they perform differently?

---

## Comparison with Analysis Pipeline

### Connection Points:

**Data Generation Output**:
```csv
model,participant,item,human_number,log_prob_0,log_prob_1,...,log_prob_N
Llama-3.1-8B,P001,1,3,-5.2,-3.1,-0.5,-2.7,...
```

**Analysis Input** (`process_survey_tasks_itemlevel.ipynb`):
```python
# Convert log probs to probabilities
data["prob_0"] = np.exp(data["log_prob_0"])
...

# Normalize to sum to 1
data["prob_pred"] = filter_pred_prob(data, human_col="human_number")

# Weight human answers by LLM probabilities
score = Σ(human_number × prob_pred) / Σ(prob_pred)
```

### The Connection:

1. **Data generation** extracts LLM logprobs for each answer option at each human answer position
2. **Analysis** converts these to probabilities and weights the HUMAN answer by the LLM probability

**This creates the circular dependency**:
- LLM sees human answer sequence during generation
- Analysis weights human answers by LLM probabilities
- Result: "LLM reliability" is actually measuring consistency of human-conditioned LLM responses

---

## Recommendations

### Immediate Actions:

1. **Document the design choice**
   - Current approach measures "human-conditional LLM consistency"
   - This is valid but different from "LLM intrinsic consistency"

2. **Quantify multi-token effects**
   - For each model, count what % of answers use multi-token numbers
   - Test if removing multi-token answers changes reliability

3. **Track data quality metrics**
   - Log how often token alignment fails
   - Report per-model statistics

### Long-term Considerations:

1. **Alternative data generation for "pure" LLM consistency**:
   - Generate LLM responses WITHOUT human context
   - Ask LLM to answer all questions independently
   - Compare LLM's own answers across items (not weighted by humans)

2. **Separate analyses**:
   - **Current method**: "LLM-human alignment" (how well LLM predicts human patterns)
   - **Alternative method**: "LLM intrinsic reliability" (LLM's own consistency)

Both are valid but answer different questions!

---

## Verdict on Data Generation

### Overall Assessment: **SOUND with caveats**

The data generation code is:
- ✅ Technically correct for extracting log probabilities
- ✅ Robust across different model types
- ✅ Well-structured and maintainable
- ⚠️ Makes a specific design choice (human-conditional) that affects interpretation
- ⚠️ Has minor technical issues (multi-token handling) that should be documented

### The design choice is not a BUG, but it does explain why:
1. The analysis naturally weights by human data (generation already uses human structure)
2. High reliability in both real and simulation (both use same human structure)
3. The methodology measures alignment rather than pure consistency

---

## Connection to Reliability Issue

The data generation is **compatible with the analysis flaw** but doesn't cause it:

- **Data generation**: Gives LLM probabilities conditional on human sequences ← Design choice
- **Analysis flaw**: Weights human answers by these probabilities AND compares across models ← Methodological error

The analysis could still be fixed to measure meaningful properties of the data:
- Within-LLM consistency given human patterns
- LLM-human alignment strength
- Variability in LLM responses

But the current analysis conflates:
- Human response structure (constant across "models")
- LLM probability distributions (variable across models)
- Random weighting effects (in simulation)

---

## Summary

**Data generation is methodologically sound** for its stated purpose: extracting LLM probabilities for human-presented questionnaires.

**However**, combined with the analysis approach, it creates a system that measures human data consistency more than LLM consistency.

**Next steps**:
1. Accept current data as-is (it's valid for certain research questions)
2. Fix the analysis to properly separate LLM signal from human structure
3. Consider generating additional data with LLM-independent responses for comparison
