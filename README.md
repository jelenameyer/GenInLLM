## In progress:
### Repository to run experiments on Reliability and Validity of using human measurement techniques to assess LLM "personality".

## How to run the experiment:
1. Go to `data_generation`.
2. Install all required packages (`requirements.txt`).
3. Run either `Calling_LLM_Models.py` with call: `python Calling_LLM_Models.py --model all --task-dir tasks` or, if OOM errors happen, run with wrapper script `run_many_models.py`.

### Caveats:
4. To run `LiquidAI/LFM2-8B-A1B` you will need this transformers version: `pip install transformers==5.0.0.dev0`.
5. For all Phi models you will need an older transformers version: `pip install transformers==4.42.3`.


### Open Data
6. All data and results of this project can be found ... (add open data location here).



## How to run the data analysis:
1. Go to `data_analysis`.
2. Build a folder `LLM_data` in which you add all data either generated or received at ... .
3. Run `data_wrangling.py` first, then `data_wrangling_behavioural_tasks.py`.
Outcome: Processed data with item responses per model.
(Format: Models function as individuals, each has answers on all subtasks of each task).




## How to generate the simulation data:
1. Go to `data_analysis`.
2.A. In the terminal, run data simulation data generation completely random: (use default settings “python simulation_data_generation.py”).
2.B. In the terminal, run data simulation data generation with offset (i.e. semi-random) “python simulation_data_generation.py --semi_random True --out_path simulation_semi_random".
3. Run `process_survey_tasks_itemlevel.py` first, then `process_behav_tasks_itemlevel.py`. Change in the code the save and read folder to `processed_data/items_per_LLM_random_simulation.csv` or `processed_data/items_per_LLM_semi_random_simulation.csv`. (The files are already there, if needed).
Outcome: Processed data with random or semi-random item responses per model.
(Format: Simulated Models function as individuals, each number has random answers on all subtasks of each task).
4. Run `simulation_data_analysis_reliability.ipynb` and see Cronbach's Alpha and Split-half Reliability for random and semi-random simulated data.
Goal: The simulation shows that not the data processing part and method lead to the good reliabilities, that we can find in the real data.
