#!/usr/bin/env python
# coding: utf-8

# # 01_Directed_Search

# **File Description**
# This file conducts a directed search across policies given a scenario. Three scenarios are defined. The scenario is determined with the case number.

# **Set up run parameters, import key packages, define model**

# In[1]:


pf_id = 2
case = 1 #define scenario here
nfe = 6#500
n_scenarios = pow(2, 1) #= 1024


# In[2]:


from ema_workbench import (ScalarOutcome, Scenario, MultiprocessingEvaluator, SequentialEvaluator)
from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress)
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.util import ema_logging

ema_logging.log_to_stderr(ema_logging.INFO)

model, planning_steps = get_model_for_problem_formulation(pf_id)


# **Define scenarios**

# In[3]:


scenarios = [Scenario('baseline', **{'discount rate 0':3,
                                       'discount rate 1':3,
                                       'discount rate 2':3,
                                       'A.1_Bmax' : 190,
                                       'A.1_Brate':1.5,
                                       'A.1_pfail':0.5,
                                       'A.2_Bmax': 190,
                                       'A.2_Brate':1.5,
                                       'A.2_pfail':0.5,
                                       'A.3_Bmax' : 190,
                                       'A.3_Brate':1.5,
                                       'A.3_pfail':0.5,
                                       'A.4_Bmax':190,
                                       'A.4_Brate':1.5,
                                       'A.4_pfail':0.5,
                                       'A.5_Bmax' : 190,
                                       'A.5_Brate':1.5,
                                       'A.5_pfail':0.5,
                                       'A.0_ID flood wave shape' :66}),
             Scenario('worstcase', **{'discount rate 0':4.5,
                                       'discount rate 1':4.5,
                                       'discount rate 2':4.5,
                                       'A.1_Bmax' : 350,
                                       'A.1_Brate':10,
                                       'A.1_pfail':1,
                                       'A.2_Bmax': 350,
                                       'A.2_Brate':10,
                                       'A.2_pfail':1,
                                       'A.3_Bmax' : 350,
                                       'A.3_Brate':10,
                                       'A.3_pfail':1,
                                       'A.4_Bmax':350,
                                       'A.4_Brate':10,
                                       'A.4_pfail':1,
                                       'A.5_Bmax' : 350,
                                       'A.5_Brate':10,
                                       'A.5_pfail':1,
                                       'A.0_ID flood wave shape' :132}),
             Scenario('bestcase', **{'discount rate 0':1.5,
                                      'discount rate 1':1.5,
                                      'discount rate 2':1.5,
                                      'A.1_Bmax' :30,
                                      'A.1_Brate':1,
                                      'A.1_pfail':0,
                                      'A.2_Bmax':30,
                                      'A.2_Brate':1,
                                      'A.2_pfail':0,
                                      'A.3_Bmax' :30,
                                      'A.3_Brate':1,
                                      'A.3_pfail':0,
                                      'A.4_Bmax':30,
                                      'A.4_Brate':1,
                                      'A.4_pfail':0,
                                      'A.5_Bmax' :30,
                                      'A.5_Brate':1,
                                      'A.5_pfail':0,
                                      'A.0_ID flood wave shape' :0})]


# **Set epsilons and run model with convergence**

# In[4]:


epsilons= [.005, .005, .005, .005, .005]


# In[5]:


convergence = [EpsilonProgress()]

# convergence = [HyperVolume(minimum=[0,0,0,0,0], maximum=[1e+08, 1e+09, 1e+10, 1000, .1]),
#                EpsilonProgress()]


# In[6]:


with MultiprocessingEvaluator(model) as evaluator:
    output = evaluator.optimize(nfe=nfe, searchover='levers', reference = scenarios[case],
                                              epsilons=epsilons,
                                              convergence=convergence)


# **Save files**

# In[7]:


experiment_number = 1


# In[8]:


import pickle
with open(f'./output/directed_search/Directed_Search_{case}_{experiment_number}.kwakkel', 'wb') as filehandler:
    pickle.dump(output, filehandler)
    filehandler.close()


# In[9]:


results, convergence = output

try:
    results.to_csv(f'./output/directed_search/Directed_Search_{case}_{experiment_number}_results.csv', index=False)
    convergence.to_csv(f'./output/directed_search/Directed_Search_{case}_{experiment_number}_convergence.csv', index=False)
except:
    print("Unable to save as csv files")


# **Load files**

# In[10]:


experiment_number = 1


# In[11]:


import pickle
with open(f'../final assignment/output/directed_search/Directed_Search_{case}_{experiment_number}.kwakkel', 'rb') as filehandler:
    output = pickle.load(filehandler)
    filehandler.close()
results, convergence = output


# **Define outcomes, generate parcoords plot, plot epsilon convergence**

# In[12]:


outcomes = results.loc[:,['Expected Annual Damage','Dike Investment Costs','RfR Investment Costs','Evacuation Costs','Expected Number of Deaths']]
len(outcomes)


# In[13]:


from ema_workbench.analysis import parcoords
import matplotlib.pyplot as plt

limits = parcoords.get_limits(outcomes)
axes = parcoords.ParallelAxes(limits, fontsize=10, rot=45)
axes.plot(outcomes, alpha=0.5)

# we invert this axis so direction of desirability is the same
#axes.invert_axis('max_P')
plt.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_paracord_scores.svg", dpi=600, bbox_inches='tight')
plt.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_paracord_scores.png", dpi=600, bbox_inches='tight')
plt.show()


# In[14]:


import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(ncols=1, sharex=True, figsize=(15,15))
ax1.plot(convergence.nfe, convergence.epsilon_progress)
ax1.set_ylabel('$\epsilon$-progress')
#ax2.plot(convergence.nfe, convergence.hypervolume)
#ax2.set_ylabel('hypervolume')

ax1.set_xlabel('number of function evaluations')
#ax2.set_xlabel('number of function evaluations')

# ax1.legend()
ax1.set_title(f"nfe vs epsilon_progress in Case {case}")

fig.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_epsilon_progress.svg", dpi=600, bbox_inches='tight')
fig.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_epsilon_progress.png", dpi=600, bbox_inches='tight')
fig.show()


# **Load files, import key packages**

# In[15]:


experiment_number = 1


# In[16]:


import pickle
with open(f'../final assignment/output/directed_search/Directed_Search_{case}_{experiment_number}.kwakkel', 'rb') as filehandler:
    output = pickle.load(filehandler)
    filehandler.close()
results, convergence = output


# In[17]:


from ema_workbench import (ScalarOutcome, Scenario, MultiprocessingEvaluator, SequentialEvaluator)
from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress)
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.util import ema_logging
import pandas as pd
import numpy as np
ema_logging.log_to_stderr(ema_logging.INFO)

model, planning_steps = get_model_for_problem_formulation(pf_id)


# **Perform experiments with random scenarios**

# In[18]:


policies = results
policies = policies.drop([o.name for o in model.outcomes], axis=1)


# In[19]:


from ema_workbench import Policy

policies_to_evaluate = []

for i, policy in policies.iterrows():
    policies_to_evaluate.append(Policy(str(i), **policy.to_dict()))


# In[20]:


with MultiprocessingEvaluator(model) as evaluator:
    results = evaluator.perform_experiments(n_scenarios,
                                            policies_to_evaluate)
experiments, outcomes = results


# **Save files**

# In[21]:


experiment_number = 2


# In[22]:


from ema_workbench import save_results
import pickle

save_results(results, f'./output/directed_search/Directed_Search_{case}_{experiment_number}.tar.gz')

with open(f'./output/directed_search/Directed_Search_{case}_{experiment_number}.kwakkel', 'wb') as filehandler:
    pickle.dump(results, filehandler)
    filehandler.close()

experiments, outcomes = results

experiments.to_csv(f'./output/directed_search/Directed_Search_{case}_{experiment_number}_exp.csv', index=False)
pd.DataFrame.from_dict(outcomes).to_csv(f'./output/directed_search/Directed_Search_{case}_{experiment_number}_out.csv', index=False)


# **Load files**

# In[23]:


experiment_number = 2


# In[24]:


import pickle
with open(f'./output/directed_search/Directed_Search_{case}_{experiment_number}.kwakkel', 'rb') as filehandler:
    output = pickle.load(filehandler)
    filehandler.close()
experiments, outcomes = output


# **Robustness: Signal to noise ratio**

# In[25]:


import seaborn as sns
def s_to_n(data, direction):
    mean = np.mean(data)
    std = np.std(data)

    if direction==ScalarOutcome.MAXIMIZE:
        return mean/std
    else:
        return mean*std


# In[26]:


import pandas as pd
import numpy as np

overall_scores = {}
for policy in np.unique(experiments['policy']):
    scores = {}

    logical = experiments['policy']==policy

    for outcome in model.outcomes:
        value  = outcomes[outcome.name][logical]
        sn_ratio = s_to_n(value, outcome.kind)
        scores[outcome.name] = sn_ratio
    overall_scores[policy] = scores
scores = pd.DataFrame.from_dict(overall_scores).T


# **Parcoords plot for signal to noise ratio**

# In[27]:


from ema_workbench.analysis import parcoords

data = scores
limits = parcoords.get_limits(data)
#limits.loc[0, ['utility', 'inertia', 'reliability', 'max_P']] = 0

paraxes = parcoords.ParallelAxes(limits, fontsize=10, rot=45)
paraxes.plot(data, alpha=0.75)
#paraxes.invert_axis('max_P')


plt.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_paracord_scores.svg", dpi=600, bbox_inches='tight')
plt.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_paracord_scores.png", dpi=600, bbox_inches='tight')
plt.show()


# **Robustness: Regret**

# In[28]:


import numpy as np
def calculate_regret(data, best):
    return np.abs(best-data)


# In[29]:


overall_regret = {}
max_regret = {}
for outcome in model.outcomes:
    policy_column = experiments['policy']

    # create a DataFrame with all the relevent information
    # i.e., policy, scenario_id, and scores
    data = pd.DataFrame({outcome.name: outcomes[outcome.name],
                         "policy":experiments['policy'],
                         "scenario":experiments['scenario']})

    # reorient the data by indexing with policy and scenario id
    data = data.pivot(index='scenario', columns='policy')

    # flatten the resulting hierarchical index resulting from
    # pivoting, (might be a nicer solution possible)
    data.columns = data.columns.get_level_values(1)

    # we need to control the broadcasting.
    # max returns a 1d vector across scenario id. By passing
    # np.newaxis we ensure that the shape is the same as the data
    # next we take the absolute value
    #
    # basically we take the difference of the maximum across
    # the row and the actual values in the row
    #
    outcome_regret = (data.max(axis=1)[:, np.newaxis] - data).abs()

    overall_regret[outcome.name] = outcome_regret
    max_regret[outcome.name] = outcome_regret.max()


# **Heatmap for regret**

# In[30]:


max_regret = pd.DataFrame(max_regret)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5,5),
                       sharey=True, sharex=True)
ax.set_title("Heatmap")
sns.heatmap(max_regret/max_regret.max(),annot_kws={'size': 10}, cmap='viridis', square=True, annot=True, ax=ax)

plt.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_heatmap.svg", dpi=600, bbox_inches='tight')
plt.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_heatmap.png", dpi=600, bbox_inches='tight')

plt.show()


# **Parcoords plot for regret**

# In[31]:


data = max_regret

colors = sns.color_palette(n_colors=len(data))


# makes it easier to identify the policy associated with each line
# in the parcoords plot
# data['policy'] = data.index.astype("float64")

limits = parcoords.get_limits(data)
#limits.loc[0, ['utility', 'inertia', 'reliability', 'max_P']] = 0

paraxes = parcoords.ParallelAxes(limits, fontsize=10, rot=45)
for i, (index, row) in enumerate(data.iterrows()):
    paraxes.plot(row.to_frame().T, label=str(index), color=colors[i], alpha=0.75)

plt.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_paracord_regret.svg", dpi=600, bbox_inches='tight')
plt.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_paracord_regret.png", dpi=600, bbox_inches='tight')

plt.show()


# **Box plots for regret**

# In[32]:


from collections import defaultdict

policy_regret = defaultdict(dict)
for key, value in overall_regret.items():
    for policy in value:
        policy_regret[policy][key] = value[policy]


# In[33]:


# this generates a 2 by 2 axes grid, with a shared X and Y axis
# accross all plots
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15,15),
                         sharey=True, sharex=True)

# to ensure easy iteration over the axes grid, we turn it
# into a list. Because there are four plots, I hard coded
# this.
axes = [axes[0,0], axes[0,1],
        axes[1,0], axes[1,1]]

# zip allows us to zip together the list of axes and the list of
# key value pairs return by items. If we iterate over this
# it returns a tuple of length 2. The first item is the ax
# the second items is the key value pair.
for ax, (policy, regret) in zip(axes, policy_regret.items()):
    data = pd.DataFrame(regret)

    # we need to scale the regret to ensure fair visual
    # comparison. We can do that by divding by the maximum regret
    data = data/max_regret.max(axis=0)
    sns.boxplot(data=data, ax=ax)

    # removes top and left hand black outline of axes
    sns.despine()

    # ensure we know which policy the figure is for
    ax.set_title(str(policy))

    ax.tick_params(axis='x', rotation=45)

plt.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_box_and_whiskers.svg", dpi=600, bbox_inches='tight')
plt.savefig(f".\output\directed_search\Directed_Search_{case}_{experiment_number}_box_and_whiskers.png", dpi=600, bbox_inches='tight')

plt.show()

