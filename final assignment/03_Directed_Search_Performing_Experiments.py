#!/usr/bin/env python
# coding: utf-8

# # 03_Directed_Search_Performing_Experiments.ipynb

# In this file we perform a large number of scenarios on the policies obtained from the previous step.

# **Imports and Setup**

# In[15]:


n_scenarios = pow(2, 1) #= 1024
pf_id = 2
name = "Rhys"


# In[16]:


from ema_workbench import (ScalarOutcome, Scenario, MultiprocessingEvaluator, SequentialEvaluator)
from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress)
from ema_workbench.analysis import parcoords
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.util import ema_logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ema_logging.log_to_stderr(ema_logging.INFO)

model, planning_steps = get_model_for_problem_formulation(pf_id)


# In[17]:


directed_search_df = pd.read_csv('output/directed_search/Directed_Search_combined_1_results.csv')


# **Re-evaluate candidate solutions under uncertainty**
# 
# We now have a large number of candidate solutions (policies), we can re-evaluate them over the various deeply uncertain factors to assess their robustness against uncertainties.
# 
# For this robustness evaluation, we perform 10000 experiments.
# 
# To reduce the run time, we will only be evaluating the best policies. We decided that those are the policies with zero deaths
# 
# 

# In[18]:


directed_narrow = directed_search_df['Expected Number of Deaths'] <= 0.0 
directed_narrow = directed_search_df[directed_narrow]
directed_narrow


# In[19]:


policies = directed_narrow.drop([o.name for o in model.outcomes], axis=1)
policies


# In[20]:


from ema_workbench import Policy

policies_to_evaluate = []

for i, policy in policies.iterrows():
    policies_to_evaluate.append(Policy(str(i), **policy.to_dict()))


# In[21]:


with MultiprocessingEvaluator(model) as evaluator:
    results = evaluator.perform_experiments(n_scenarios,
                                            policies_to_evaluate)


# In[30]:


outcomes = results[1]
experiments = results[0]


# In[33]:


outcomes_df = pd.DataFrame.from_dict(outcomes)


# In[34]:


outcomes_df.to_csv(f'./output/directed_search/Directed_Search_Scenario_Analysis_Outcomes_{name}.csv', index=True)
experiments.to_csv(f'./output/directed_search/Directed_Search_Scenario_Analysis_Experiments_{name}.csv', index=True)


# In[35]:


directed_narrow.to_csv(f'./output/directed_search/Directed_Search_Scenario_Analysis_Policy_{name}.csv', index=True)


# In[ ]:




