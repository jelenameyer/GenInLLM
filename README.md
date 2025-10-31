## In progress:
### Repository to run experiments on Reliability and Validity of using human measurement techniques to assess LLM "personality".

## How to run the experiment:
1. Go to `data_generation`.
2. Install all required packages (`requirements.txt`).
3. Run either `Calling_LLM_Models.py` with call: `python Calling_LLM_Models.py --model all --task-dir tasks` or, if OOM errors happen, run with wrapper script `run_many_models.py`.

### Caveats:
4. To run `LiquidAI/LFM2-8B-A1B` you will need this transformers version: `pip install transformers==5.0.0.dev0`.
5. For all Phi models you will need an older transformers version: `pip install transformers==4.42.3`.
