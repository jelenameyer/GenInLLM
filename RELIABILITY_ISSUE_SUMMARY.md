# Critical Reliability Issue - Executive Summary

**Date:** November 22, 2025
**Issue:** Fundamental methodological flaw in reliability analysis

## TL;DR

**The high reliability scores reported for LLM "personality" assessments are artifacts of the methodology, not genuine LLM consistency.**

Specifically:
- ~25% of reported reliabilities (including the highest ones) are indistinguishable from random noise
- The simulation control inadvertently reveals this flaw by showing identical reliability for random data
- Root cause: The analysis weights human participant data by LLM probabilities, allowing human data structure to dominate the signal

## The Problem in One Sentence

The study measures **"consistency of weighted averages of human decisions"** instead of **"consistency of LLM responses"**.

## Key Evidence

### Smoking Gun: Identical Reliability in Real vs. Random Data

| Task | Real LLM Data α | Random Simulation α |
|------|-----------------|---------------------|
| BART | 0.985 | 0.985 |
| AUDIT | 0.740 | 0.740 |
| BARRAT (all subscales) | 0.76-0.83 | 0.76-0.83 |
| CARE (all subscales) | 0.65-0.96 | 0.65-0.96 |

**Interpretation:** If random log probabilities produce the same reliability as real LLM responses, the reliability is not measuring LLM behavior.

## Root Cause

### Current (Flawed) Methodology:

```
1. Collect LLM log probabilities for answer options
2. Match to HUMAN participant decisions  ← Problem starts here
3. Weight human decisions by LLM probabilities
4. Aggregate across thousands of humans per item
5. Compute reliability across items
6. Result: Measures stability of human data structure
```

### What Gets Preserved in "Random" Simulation:

**Randomized:**
- LLM log probabilities ✓

**Preserved (FLAW):**
- All human participant IDs
- All human decisions/responses
- Complete data structure
- Grouping by participant × item

**Result:** "Random" simulation still contains all the human data structure, just weighted differently.

## Why This Went Undetected

1. **Complex pipeline** obscures the issue across multiple processing steps
2. **Simulation control** appears to validate findings (but actually reveals the flaw)
3. **Some tasks differ** (CCT, DFD) suggesting the method works for those
4. **Matches intuitions** about LLM capabilities
5. **Technically correct** steps (proper Cronbach's α calculation, etc.)

The flaw is **conceptual** not computational.

## Impact Assessment

### Affected Results:

- **BART (α=0.985)**: Highest reliability reported → ARTIFACT
- **CARE "warmth" (α=0.961)**: Second highest → ARTIFACT
- **AUDIT, BARRAT subscales (α=0.74-0.83)**: → ARTIFACTS
- **~8 out of 33** task/subscale combinations show identical real/simulation reliability

### Unaffected Results:

Tasks with large real vs. simulation differences may still be valid:
- CCT (real: 0.985, sim: -0.233) → Likely genuine
- DFD/DFE tasks → Likely genuine
- DOSPERT domains → Need further investigation

## Recommended Actions

### Immediate:

1. **Acknowledge the issue** in any presentations/publications using these results
2. **Reanalyze data** using direct LLM responses (not human-weighted)
3. **Revise claims** about reliability magnitude

### Short-term:

1. **Implement corrected methodology** (see detailed report)
2. **Re-run analysis** on subset of tasks to validate correction
3. **Compare outcomes** between original and corrected approaches
4. **Document lessons learned** for the field

### Long-term:

1. **Test-retest reliability** by re-running LLMs on same items
2. **Within-LLM analysis** (each model's internal consistency)
3. **Validation studies** with independent methods

## Corrected Methodology (Brief)

Instead of:
```python
score = Σ(human_decision × LLM_prob) / Σ(LLM_prob)
```

Use:
```python
score = argmax(LLM_prob)  # Direct LLM response
# OR
score = Σ(answer_value × LLM_prob)  # Expected value from LLM only
```

Key difference: **No human data in the scoring process**.

## Where to Go From Here

1. **Read full analysis**: `data_analysis/reliability_analysis_findings.md`
2. **Review methodology**: `data_analysis/process_*_itemlevel.ipynb`
3. **Check simulation**: `data_analysis/random_simulation_data_generation.py`
4. **See test script**: `data_analysis/test_simulation_mechanism.py`

## Questions?

This issue is complex and the pipeline involves many steps. The full technical report provides:
- Detailed explanation of the mechanism
- Mathematical analysis
- Specific code locations
- Proposed corrections
- Expected outcomes of corrected analysis

---

**Bottom line:** High reliability scores in BART, AUDIT, BARRAT, and CARE are methodological artifacts. True LLM reliability must be re-assessed using corrected methodology that does not inject human data structure into the measurement.
