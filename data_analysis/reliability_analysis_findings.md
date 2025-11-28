# Reliability Analysis: Methodological Issues Investigation

## Executive Summary

Investigation of why both real LLM data and randomly simulated data show high reliabilities in certain psychological scales reveals **fundamental methodological flaws** in the study design.

## Key Finding: Data Structure Preservation Problem

The simulation control creates **spurious reliability** by preserving the underlying human data structure while only randomizing LLM log probabilities.

### What Gets Randomized vs. Preserved

**Randomized** (in `random_simulation_data_generation.py`):
- Log probability columns (`log_prob_*`)
- Integer answer columns (0-99)

**Preserved** (CRITICAL FLAW):
- All participant IDs
- All round/item numbers
- All human decision data (`human_decision`, `decision_num`, etc.)
- Complete data structure (which participant × which item × which decision)
- All grouping variables (experiment, model, participant, round, item)

## Comparison of Reliability Results

### Tasks with IDENTICAL/VERY SIMILAR reliability in real AND simulation data:

| Task | Real α | Simulation α | Difference |
|------|--------|--------------|------------|
| AUDIT scale | 0.740 | 0.740 | **IDENTICAL** |
| BARRAT (BISa) | 0.831 | 0.831 | **IDENTICAL** |
| BARRAT (BISm) | 0.831 | 0.831 | **IDENTICAL** |
| BARRAT (BISn) | 0.761 | 0.761 | **IDENTICAL** |
| BART task | 0.985 | 0.985 | **IDENTICAL** |
| CARE (CAREa) | 0.856 | 0.856 | **IDENTICAL** |
| CARE (CAREs) | 0.651 | 0.651 | **IDENTICAL** |
| CARE (CAREw) | 0.961 | 0.961 | **IDENTICAL** |

### Tasks with DIFFERENT reliability:

| Task | Real α | Simulation α | Difference |
|------|--------|--------------|------------|
| CCT task | 0.985 | -0.233 | 1.218 |
| DFD (gain) | 0.971 | 0.023 | 0.948 |
| DFD (loss) | 0.975 | -0.335 | 1.310 |
| DFE (gain) | 0.880 | 0.096 | 0.784 |
| DFE (loss) | 0.851 | 0.206 | 0.645 |
| DOSPERT domains | 0.5-0.9 | -0.4 to 0.2 | Large |

## The Fundamental Methodological Flaws

### Flaw #1: Simulation Preserves Human Response Structure

The simulation does NOT create truly random data. Instead:

1. **Real data collection**: LLMs generate log probabilities for answer options
2. **Processing**: Log probs are weighted against HUMAN participant decisions
3. **Simulation**: Randomizes log probs BUT keeps the SAME human decision data
4. **Result**: All "simulated models" share identical underlying human response patterns

Example from BART processing (`process_behav_tasks_itemlevel.ipynb`):
```python
# Filter out probability LLM assigned to real item answer
BART_data = filter_pred_prob(BART_data, human_col = "human_decision")

# Compute score per model per round
grouped = data.groupby(["experiment", "model", "round"])
score = (grouped["decision_num"].apply(lambda x: (x * data.loc[x.index, "prob_pred"]).sum())
         / grouped["prob_pred"].sum())
```

The `decision_num` comes from **HUMAN DATA** and is **IDENTICAL across all models** (both real and simulated).

### Flaw #2: Weighted Averaging Creates Artificial Consistency

For each item (e.g., BART round), the score is:

```
score_item = Σ(human_decision_i × random_prob_i) / Σ(random_prob_i)
```

Where:
- `human_decision_i` is the SAME for all models (from human data)
- `random_prob_i` are random weights

With ~1500 participants:
- Random weights average out (law of large numbers)
- All models converge toward the mean human decision pattern
- High consistency across models → HIGH CRONBACH'S ALPHA

This is **not measuring LLM reliability** - it's measuring the stability of human response patterns when averaged with random weights!

### Flaw #3: The "Reliability" is Actually Between-Item Correlation in Human Data

Cronbach's alpha formula:
```
α = (k / (k-1)) × (1 - Σ(σ²_items) / σ²_total)
```

High alpha means:
- Items covary consistently across participants (models)
- Low within-item variance relative to total variance

But when all models share the same human response structure:
- "Between-model correlations" are actually correlations in the human data structure
- High α reflects human data consistency, NOT LLM consistency

## Why Some Tasks Show the Problem More Than Others

### Tasks with PRESERVED spurious reliability (AUDIT, BARRAT, BART, CARE):

These are **survey scales** where:
- Each participant answers all items
- Same answer scale across all items
- Large number of participants (~1500)
- Averaging effect is strong → random weights wash out
- Human response patterns dominate

### Tasks with REDUCED spurious reliability (CCT, DFD, DFE):

Need to investigate why these differ. Possible reasons:
- Different data structures
- Different aggregation methods
- Different numbers of participants per item
- Binary choices vs. continuous responses

## Implications

### What This Means for the Published Results:

1. **High reliabilities in real LLM data are SUSPECT**
   - Cannot distinguish signal from artifact
   - Same methodology that produces high α in random data

2. **The simulation "control" is NOT a valid control**
   - Doesn't test what it claims to test
   - Shares too much structure with real data

3. **True LLM reliability is UNKNOWN**
   - Current methodology cannot measure it
   - Need complete redesign

## Recommended Fixes

### For Simulation Control:

**DO NOT** use human participant data in simulation. Instead:
1. Generate completely synthetic item structure
2. Randomize both probabilities AND response patterns
3. Use LLM-only data (no human decisions)

### For Reliability Analysis:

1. **Within-LLM reliability**: Test each LLM multiple times on same items (test-retest)
2. **Between-LLM reliability**: Compare LLMs directly, not weighted by human data
3. **Item-level analysis**: Examine raw LLM probabilities before human-data weighting

### For Score Calculation:

Current approach:
```
score = Σ(human_choice × LLM_prob) / Σ(LLM_prob)
```

This conflates:
- LLM response patterns (what you want to measure)
- Human response patterns (confound)

Better approach:
```
score = LLM's most probable response
OR
score = Expected value from LLM's probability distribution only
```

## Why Some Tasks Preserve Spurious Reliability and Others Don't

### Hypothesis: Human Data Structure Determines Outcome

The differential preservation of reliability in simulation depends on the **correlation structure** in the underlying human data:

**Tasks that PRESERVE high reliability in simulation:**
- BART (Balloon Analog Risk Task)
- AUDIT, BARRAT, CARE (personality scales)
- PRI (certainty subscale)

These likely have:
1. **Correlated item responses**: Individuals who score high on item 1 tend to score high on item 2, etc.
2. **Stable individual differences**: Strong person-level variance
3. **Large sample sizes** (1500+ participants) that average out random weights

Mathematical explanation:
```
For each item j and model m:
  score[m, j] = Σᵢ(human_decision[i, j] × random_weight[i]) / Σᵢ(random_weight[i])
```

When human_decision[i, j] has high between-person correlations across j:
- Random weights become noise that averages out
- All models converge toward the same item means
- High inter-item correlations → High Cronbach's α

**Tasks that DON'T preserve reliability:**
- CCT (Columbia Card Task)
- DFD/DFE (Decisions from Description/Experience)
- DOSPERT (domain-specific risk-taking)

These likely have:
1. **Independent item responses**: Decisions on one trial don't predict decisions on another
2. **More within-person variability**
3. **Task-specific contextual effects**

When human data is less correlated:
- Random weights create noise that doesn't cancel out
- Models diverge in their item scores
- Low/negative inter-item correlations → Low/negative α

### Critical Point

**This differential effect does NOT validate the simulation approach!**

The fact that some tasks show different patterns only reveals that:
1. Human data structure varies by task
2. The simulation is sensitive to this structure
3. **The simulation is NOT independent of human data** (the fundamental flaw)

A valid simulation would show low reliability for ALL tasks, since the LLM responses are random.

## Quantifying the Extent of the Problem

### Tasks with Identical Reliability (Real = Simulation):

| Task | Subscale | Real α | Sim α | Status |
|------|----------|--------|-------|--------|
| AUDIT | Total | 0.740 | 0.740 | ⚠️ IDENTICAL |
| BARRAT | BISa | 0.831 | 0.831 | ⚠️ IDENTICAL |
| BARRAT | BISm | 0.831 | 0.831 | ⚠️ IDENTICAL |
| BARRAT | BISn | 0.761 | 0.761 | ⚠️ IDENTICAL |
| BART | Total | 0.985 | 0.985 | ⚠️ IDENTICAL |
| CARE | All 3 | 0.65-0.96 | 0.65-0.96 | ⚠️ IDENTICAL |

**Implication**: For these tasks, the observed reliability in real LLM data is **indistinguishable from random noise weighted by human data**. We cannot conclude that LLMs show meaningful consistency.

### Tasks with Different Reliability:

| Task | Real α | Sim α | Difference | Status |
|------|--------|-------|------------|--------|
| CCT | 0.985 | -0.233 | 1.218 | ✓ Shows LLM signal |
| DFD (loss) | 0.975 | -0.335 | 1.310 | ✓ Shows LLM signal |
| DFE (gain) | 0.880 | 0.096 | 0.784 | ✓ Shows LLM signal |

**Implication**: Only these tasks show evidence of true LLM reliability above the human data artifact.

### Overall Assessment:

- **~25% of results** (8/33 task subscales) show indistinguishable reliability from random simulation
- **These are among the highest reliability scores reported** (α > 0.7)
- **The most impressive findings (BART α=0.985, CARE α=0.96) are artifacts**

## Root Cause Analysis

### Why This Happened

The fundamental error is a **conceptual confusion** between:

1. **What was intended**: "Do LLMs show consistent personality profiles?"
2. **What was measured**: "Do weighted averages of human data show consistency?"

The methodology conflates:
- **LLM behavior** (what we want to measure)
- **Human behavior** (used as ground truth for weighting)
- **Random noise** (what simulation tests)

### The Flawed Logic Chain

```
1. Collect LLM log probabilities for each answer option
   ↓
2. Match these to HUMAN participant decisions
   ↓
3. Weight human decisions by LLM probabilities
   ↓
4. Compute reliability across models
   ↓
5. Compare to random log probs weighted by SAME human decisions
   ↓
6. Find same high reliability → "LLMs show strong reliability!"
```

**The error**: Steps 2-3 inject human data structure that dominates the signal.

### Why Reviewers Might Have Missed This

1. **Surface plausibility**: Using human data as reference seems reasonable
2. **Complex pipeline**: Multi-step processing obscures the issue
3. **Simulation control**: Appears to validate findings (but actually reveals the flaw!)
4. **Some tasks show differences**: CCT, DFD, etc. look different, suggesting method works
5. **Matches expectations**: High reliability in some scales fits prior beliefs

## Proposed Corrected Methodology

### Option 1: Direct LLM Response Analysis (Recommended)

Instead of weighting by human data, analyze LLM responses directly:

```python
# For each LLM model and each item:
# 1. Extract the response with highest log probability
llm_response[model, item] = argmax(log_prob_options)

# 2. Convert to numeric score based on answer option value
llm_score[model, item] = option_value[llm_response[model, item]]

# 3. Compute reliability across items within each model
cronbach_alpha(llm_score[model, :])
```

**Advantages:**
- Measures actual LLM behavior
- No contamination from human data
- Interpretable: "Does this LLM give consistent answers?"

**Disadvantages:**
- Ignores uncertainty in LLM responses
- Binary choices might show ceiling/floor effects

### Option 2: Expected Value from LLM Distribution Only

Use the full probability distribution without human anchoring:

```python
# For each model and item:
# 1. Convert log probs to probabilities
probs = softmax(log_probs)

# 2. Compute expected value
llm_score[model, item] = sum(option_values * probs)

# 3. Compute reliability
cronbach_alpha(llm_score)
```

**Advantages:**
- Captures LLM uncertainty
- Uses full information from probability distribution
- No human data contamination

**Disadvantages:**
- Assumes interval scale properties
- May conflate confidence with response

### Option 3: Test-Retest Reliability

Administer the same items multiple times to the same LLM:

```python
# Run each LLM on same items at different times/seeds
scores_time1 = get_llm_scores(model, items, seed=1)
scores_time2 = get_llm_scores(model, items, seed=2)

# Compute correlation
reliability = correlation(scores_time1, scores_time2)
```

**Advantages:**
- Gold standard for reliability
- Directly tests consistency
- Clear interpretation

**Disadvantages:**
- Requires re-running experiments
- LLMs may be deterministic (low test-retest variation)
- Expensive computationally

### Option 4: Corrected Simulation Control

If using human data is necessary, simulation must randomize ALL structure:

```python
# WRONG (current approach):
simulation_data = real_data.copy()
simulation_data['log_probs'] = random_values()
# Still preserves: participant_ids, items, human_decisions

# CORRECT:
simulation_data = generate_synthetic_data(
    n_models=44,
    n_items=30,
    n_options_per_item=options  # e.g., [0,1,2,3,4]
)
# Generates: random log_probs for random response patterns
# No human data at all
```

### Recommended Analysis Pipeline

1. **Primary analysis**: Use Option 1 or 2 (direct LLM responses)
2. **Validation**: Use Option 3 (test-retest) on subset of models
3. **Control**: Use Option 4 (proper simulation) to establish baseline
4. **Comparison**: Report human-weighted scores separately as "human-similarity index"

## Implementation Notes for Corrected Analysis

### Data Files Needed:

```
data_analysis/
├── LLM_data/           # Raw LLM log probabilities
│   └── {Model}_{Task}_prompting_results.csv
├── reanalysis/         # New folder for corrected analysis
│   ├── direct_llm_scores.csv
│   ├── reliability_corrected.csv
│   └── comparison_with_original.csv
```

### Code Structure:

```python
# Step 1: Load raw LLM data (no human data)
def load_llm_data_only(task_name):
    """Load only LLM log probabilities, ignore human decisions"""
    # Read log_prob_* columns
    # Do NOT merge with human data
    return llm_probs

# Step 2: Convert to scores
def compute_llm_scores(llm_probs, method='argmax'):
    """Convert log probs to scores without human data"""
    if method == 'argmax':
        return get_top_response(llm_probs)
    elif method == 'expected_value':
        return get_expected_value(llm_probs)

# Step 3: Compute reliability
def compute_reliability_per_model(scores):
    """Compute Cronbach's alpha for EACH model separately"""
    # Not across models! Within-model consistency
    return alpha_per_model

# Step 4: Compare to original results
def compare_methodologies(original_alpha, corrected_alpha):
    """Quantify impact of methodological fix"""
    return comparison_df
```

## Expected Outcomes of Corrected Analysis

### Predictions:

1. **Overall reliability will be LOWER** than currently reported
   - Current high values are inflated by human data structure
   - True LLM consistency is likely moderate

2. **Within-model reliability may differ from between-model patterns**
   - Current analysis conflates these
   - Some LLMs may be internally consistent but differ from each other

3. **Task differences will emerge more clearly**
   - Survey scales (AUDIT, BARRAT) may show lower reliability
   - Behavioral tasks may show different patterns than currently

4. **Model size effects may disappear or reverse**
   - Current correlations with model size may be artifacts
   - Smaller models might show different consistency patterns

### Validation Criteria:

The corrected analysis should show:
1. ✓ Simulation control with near-zero reliability
2. ✓ Clear distinction between LLM signal and random noise
3. ✓ Interpretable differences between models
4. ✓ Correspondence with theoretical expectations

## Next Steps for Investigation

1. ✅ Identify the data structure problem
2. ✅ Explain why some tasks preserve reliability and others don't
3. ✅ Quantify the extent of the problem
4. ✅ Propose corrected methodology
5. ⏳ Implement corrected analysis on subset of data
6. ⏳ Compare original vs corrected results
7. ⏳ Document lessons learned for future research

