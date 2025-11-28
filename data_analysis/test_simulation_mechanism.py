"""
Empirical test to understand why some tasks preserve high reliability in simulation
while others don't.

Hypothesis testing:
1. Check if normalized random probabilities cluster around 0.5
2. Examine variance structure in human decision data
3. Compare BART vs CCT data characteristics
"""

import numpy as np
import pandas as pd
from scipy import stats

# Simulate the randomization process
np.random.seed(42)
n_samples = 10000

# Generate random log probabilities like in the simulation
log_prob_A = np.random.uniform(-20, -1, n_samples)
log_prob_B = np.random.uniform(-20, -1, n_samples)

# Normalize to probabilities (like in the processing code)
prob_A = np.exp(log_prob_A) / (np.exp(log_prob_A) + np.exp(log_prob_B))
prob_B = np.exp(log_prob_B) / (np.exp(log_prob_A) + np.exp(log_prob_B))

print("="*70)
print("ANALYSIS OF RANDOM PROBABILITY NORMALIZATION")
print("="*70)
print(f"\nRandom log_prob range: -20 to -1")
print(f"Number of samples: {n_samples}")
print(f"\nNormalized probability A statistics:")
print(f"  Mean: {prob_A.mean():.4f}")
print(f"  Std:  {prob_A.std():.4f}")
print(f"  Min:  {prob_A.min():.4f}")
print(f"  Max:  {prob_A.max():.4f}")
print(f"\nProbability B statistics (should be complement):")
print(f"  Mean: {prob_B.mean():.4f}")
print(f"  Std:  {prob_B.std():.4f}")

# Check if sum equals 1 (sanity check)
print(f"\nSum of probabilities (should be 1.0): {(prob_A + prob_B).mean():.10f}")

# Distribution characteristics
print(f"\nDistribution shape:")
print(f"  Proportion near 0.5 (0.4-0.6): {((prob_A > 0.4) & (prob_A < 0.6)).mean():.2%}")
print(f"  Proportion extreme (<0.2 or >0.8): {((prob_A < 0.2) | (prob_A > 0.8)).mean():.2%}")

print("\n" + "="*70)
print("WEIGHTED AVERAGING SIMULATION")
print("="*70)

# Simulate what happens when we weight human decisions with random probabilities
# Assume we have 1500 participants with varying human decisions per item

n_participants = 1500
n_models = 44
n_items = 30

# Scenario 1: Human decisions are CORRELATED across items (like BART might be)
# People who pump a lot on item 1 also tend to pump a lot on item 2, etc.
print("\nScenario 1: CORRELATED human decisions across items")
print("-" * 70)

# Create correlated human data: base tendency + item-specific variation
human_tendency = np.random.normal(35, 8, n_participants)  # Each person's base tendency
item_effects = np.random.normal(0, 2, n_items)  # Item difficulty

# Human decisions: person tendency + item effect + noise
human_data_correlated = np.zeros((n_participants, n_items))
for i in range(n_participants):
    for j in range(n_items):
        human_data_correlated[i, j] = max(0, human_tendency[i] + item_effects[j] + np.random.normal(0, 3))

# Simulate multiple models with random weights
model_scores_correlated = np.zeros((n_models, n_items))
for m in range(n_models):
    for item in range(n_items):
        # Random weights (like random log probs converted to probs)
        random_weights = np.random.uniform(0, 1, n_participants)

        # Weighted average of human decisions
        weighted_sum = np.sum(human_data_correlated[:, item] * random_weights)
        weight_sum = np.sum(random_weights)
        model_scores_correlated[m, item] = weighted_sum / weight_sum

# Calculate Cronbach's alpha manually
df_corr = pd.DataFrame(model_scores_correlated)
item_vars = df_corr.var(axis=0).sum()
total_var = df_corr.sum(axis=1).var()
k = n_items
alpha_correlated = (k / (k - 1)) * (1 - item_vars / total_var)

print(f"Cronbach's alpha with correlated human data: {alpha_correlated:.4f}")
print(f"Mean item correlation: {df_corr.corr().values[np.triu_indices_from(df_corr.corr().values, k=1)].mean():.4f}")

# Scenario 2: Human decisions are INDEPENDENT across items (like CCT might be)
print("\nScenario 2: INDEPENDENT human decisions across items")
print("-" * 70)

# Each item has independent human decisions
human_data_independent = np.random.normal(35, 8, (n_participants, n_items))
human_data_independent = np.maximum(0, human_data_independent)  # No negative decisions

# Simulate multiple models with random weights
model_scores_independent = np.zeros((n_models, n_items))
for m in range(n_models):
    for item in range(n_items):
        # Random weights
        random_weights = np.random.uniform(0, 1, n_participants)

        # Weighted average of human decisions
        weighted_sum = np.sum(human_data_independent[:, item] * random_weights)
        weight_sum = np.sum(random_weights)
        model_scores_independent[m, item] = weighted_sum / weight_sum

# Calculate Cronbach's alpha
df_ind = pd.DataFrame(model_scores_independent)
item_vars_ind = df_ind.var(axis=0).sum()
total_var_ind = df_ind.sum(axis=1).var()
alpha_independent = (k / (k - 1)) * (1 - item_vars_ind / total_var_ind)

print(f"Cronbach's alpha with independent human data: {alpha_independent:.4f}")
print(f"Mean item correlation: {df_ind.corr().values[np.triu_indices_from(df_ind.corr().values, k=1)].mean():.4f}")

print("\n" + "="*70)
print("KEY INSIGHT")
print("="*70)
print("""
The simulation preserves high reliability when the human data has:
1. Correlated structure across items (individual differences)
2. Large sample size that averages out random weights

The simulation shows LOW/negative reliability when:
1. Human data is more independent across items
2. Random weights don't preserve a consistent structure

This explains why BART/CARE (personality-like traits) preserve high alpha
but behavioral tasks with more independent trials (CCT?) show low alpha.

HOWEVER, this doesn't change the fundamental flaw: the simulation is NOT
testing what it claims to test. It's still contaminated by human data structure.
""")
