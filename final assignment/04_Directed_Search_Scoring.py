#!/usr/bin/env python
# coding: utf-8

# # 04_Directed_Search_Scoring.ipynb

# In this final file we calculate the different robustness scores to select a final policy set.

# In[19]:


pf_id = 2
name = 'Rhys'


# In[20]:


from ema_workbench import (ScalarOutcome, Scenario, MultiprocessingEvaluator, SequentialEvaluator)
from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress)
from ema_workbench.analysis import parcoords
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.util import ema_logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


ema_logging.log_to_stderr(ema_logging.INFO)

model, planning_steps = get_model_for_problem_formulation(pf_id)


# In[21]:


outcomes = pd.read_csv('output/directed_search/Directed_Search_Scenario_Analysis_Rhys_out.csv', index_col = 0)
experiments = pd.read_csv('output/directed_search/Directed_Search_Scenario_Analysis_Rhys_exp.csv', index_col = 0)
policies = pd.read_csv('output/directed_search/Directed_Search_Scenario_Analysis_Policy_Rhys.csv', index_col = 0)


# In[22]:


results = pd.concat([experiments, outcomes], axis = 1)
results


# We can now evaluate the **robustness** of each of the policy options based on these scenario results. 
# There are multiple metrics to quantify robustness. On of them is the *signal to noise ratio*, which is simply the mean of a dataset divided by its standard deviation. For instance, for an outcome indicator to be maximized, we prefer a high average value across the scenarios, and a low standard deviation, implying a narrow range of uncertainty about the outcomes. Therefore, we want to maximize the signal-to-noise ratio. For an outcome indicator to be minimized, a lower mean and a lower standard deviation is preferred. Therefore the formulation should be different. (Copied from J Kwakkel's labs)
# 

# In[23]:


def s_to_n(data, direction):
    mean = np.mean(data)
    std = np.std(data)
    if std < 10e-5:
        std = 0
    if direction==ScalarOutcome.MAXIMIZE:
        return mean/std
    else:
        return mean*std
    


# In[24]:


x = results.where(results["policy"] == 285)
x = x.dropna()
x
# print(x['Dike Investment Costs'])
# sn_ratio = s_to_n(x['Dike Investment Costs'], 'MINIMIZE')

# print(sn_ratio)
# print(np.mean(x['Dike Investment Costs']))
# int(np.std(x['Dike Investment Costs']))
# std = np.std(x["Expected Number of Deaths"])
# mean = np.mean(x["Expected Number of Deaths"])
# print(std,mean)
# print(std*mean)
# x


# In[25]:


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
scores = scores.drop(['RfR Investment Costs','Dike Investment Costs'], axis=1)
normalized_scores=(scores-scores.min())/(scores.max()-scores.min())


# In[26]:


data = normalized_scores
limits = parcoords.get_limits(data)
limits.loc[0, ['Expected Annual Damage',  'Evacuation Costs','Expected Number of Deaths']] = 0
plt.rcParams["figure.figsize"] = (15,10)
paraxes = parcoords.ParallelAxes(limits)
paraxes.plot(data)
paraxes.legend()
paraxes.invert_axis(['Expected Annual Damage', 'Evacuation Costs','Expected Number of Deaths'])

plt.show()


# We prioritize in the order of deaths (low) > damage (low) > dike (High) (transport company likes dikes).

# In[27]:


normalized_scores


# Another robustness metric is **maximum regret**, calculated again for each policy and for each outcome indicator. *Regret* is defined for each policy under each scenario, as the difference between the performance of the policy in a specific scenario and the berformance of a no-regret (i.e. best possible result in that scenario) or reference policy. The *maximum regret*  is then the maximum of such regret values across all scenarios. We of course favor policy options with low *maximum regret* values. 
# 
# **Write a function to calculate the maximum regret for both kinds of outcome indicators. Calculate the maximum regret values for each outcome and each policy option. Plot the tradeoffs on a parallel plot. Which solutions look like a good compromise policy?**

# In[28]:


def calculate_regret(data, best):
    return np.abs(best-data)


# Regret is the performance difference between the best possible outcome in a scenario across policies, the the observed outcome for a given policy. We have in this case both minimization and maximization. Best means the lowest in case of mimimization, and heighest in case of maximization. To avoid having to explicitly account for this in how we calculate the difference, we can simply take the absolute value of the difference. In this case, max_P will return negative regret values for `best-data`, so by taking the absolute value, we fix this
# 
# The next part of the code is probably the most tricky part. We need to find the best possible outcome for each scenario. We could do this by iterating over the scenario_id column in the experiment array. But we can also use pandas instead as done below. Wat we do is the following:
# 1. we create a dataframe with the outcome, the name of the policy and the scenario. This is a so called long-form representation of the data
# 2. We want to have the results for each policy side by side so we can take the max, or min accross the column. The pivot method on the DataFrame does this for us
# 3. We take the maximum or minimum accross the row.

# In[29]:


# experiments, outcomes = results

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
    outcome_regret = (data.min(axis=1)[:, np.newaxis] - data).abs()
    
    overall_regret[outcome.name] = outcome_regret
    max_regret[outcome.name] = outcome_regret.max()
    


# In[30]:


max_regret = pd.DataFrame(max_regret)
sns.heatmap(max_regret/max_regret.max(), cmap='viridis', annot=True)
plt.show()


# In[31]:


max_regret_normalized = (max_regret-max_regret.min())/(max_regret.max()-max_regret.min())
max_regret_normalized


# In[32]:


def getColor(c, N, idx):
    import matplotlib as mpl
    cmap = mpl.cm.get_cmap(c)
    norm = mpl.colors.Normalize(vmin=0.0, vmax=N - 1)
    return cmap(norm(idx))

data = max_regret_normalized

# makes it easier to identify the policy associated with each line
# in the parcoords plot
# data['policy'] = data.index.astype("float64")

limits = parcoords.get_limits(data)
limits.loc[0, ['Expected Annual Damage', 'Dike Investment Costs', 'Evacuation Costs','Expected Number of Deaths']] = 0
paraxes = parcoords.ParallelAxes(limits)

for i, (index, row) in enumerate(data.iterrows()):
    paraxes.plot(row.to_frame().T, label=str(index), color=getColor('magma',len(data),i))
paraxes.legend()
    
plt.show()


# In[33]:


from collections import defaultdict

policy_regret = defaultdict(dict)
for key, value in overall_regret.items():
    for policy in value:
        policy_regret[policy][key] = value[policy]


# In[34]:


# this generates a 2 by 2 axes grid, with a shared X and Y axis
# accross all plots
fig, axes = plt.subplots(ncols=3, nrows=4, figsize=(40,10), 
                         sharey=True, sharex=True)

# to ensure easy iteration over the axes grid, we turn it
# into a list. Because there are four plots, I hard coded
# this. 
axes = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2],
        axes[2,0], axes[2,1], axes[2,2], axes[3,0], axes[3,1], axes[3,2]]
# zip allows us to zip together the list of axes and the list of 
# key value pairs return by items. If we iterate over this
# it returns a tuple of length 2. The first item is the ax
# the second items is the key value pair.

for ax, (policy, regret) in zip(axes, policy_regret.items()):
    print(policy)
    data = pd.DataFrame(regret)

    # we need to scale the regret to ensure fair visual
    # comparison. We can do that by divding by the maximum regret
    data = data/max_regret.max(axis=0)
    sns.set(rc={'figure.figsize':(100,100)})
    sns.boxplot(data=data, ax=ax)
    
    # removes top and left hand black outline of axes
    sns.despine()
    
    # ensure we know which policy the figure is for
    ax.set_title(str(policy))


plt.show()


# In[35]:


fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(40,10), 
                         sharey=True, sharex=True)


axes = [axes[0], axes[1], axes[2]]
policies = [410,437,705]
policy_items = {}
for i in policies:
    policy_items[i] = policy_regret[i]

for ax, (policy, regret) in zip(axes, policy_items.items()):

    data = pd.DataFrame(regret)

    # we need to scale the regret to ensure fair visual
    # comparison. We can do that by divding by the maximum regret
    data = data/max_regret.max(axis=0)
    sns.set(rc={'figure.figsize':(100,100)})
    sns.boxplot(data=data, ax=ax)
    
    # removes top and left hand black outline of axes
    sns.despine()
    
    # ensure we know which policy the figure is for
    ax.set_title(str(policy))


plt.show()


# This is in line with the maximum regret parallel coordinates plot, but we get some more details.

# We now have an understanding of which solutions have decent robustness using 2 different robustness metrics. 
# 

# In[36]:


results.to_csv(f'./output/directed_search/Directed_Search_Scenario_Analysis_{name}.csv', index=False)

