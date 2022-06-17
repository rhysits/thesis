# EPA1361-G1 README File

Created by: EPA1361 Group 7

|        Name        | Student Number |
|:------------------:|:---------------|
|  Madison Berry     | 5634431        | 
| Michiel van Dalsum | 4647556        |
|     Rhys Evans     | 5633273        |
|    Morris Huang    | 5487781        |
|   Matvei Isaenko   | 5618053        |
|    Bo de Vries     | 4728823        |


## Introduction

The report can be found at [final_assignment/report/EPA1361_MBDM_G1_Report.pdf](final_assignment/report/EPA1361_MBDM_G1_Report.pdf) in the [report](report) directory.

## How to Use

To run the model processes, navigate to the following Jupyter Notebook files in the [final assignment](final_assignment/) directory:
* [00_0_Open_Exploration_Scenarios.ipynb](final_assignment/00_0_Open_Exploration_Scenarios.ipynb): This file conducts open exploration across scenarios given a policy where no levers are applied.
* 
* [00_1_Open_Exploration_Levers.ipynb](final_assignment/00_0_Open_Exploration_Levers.ipynb): This file conducts open exploration across the levers given a baseline scenario. The baseline scenario is defined as the average of the range for each uncertainty.

* [01_0_Directed_Search.ipynb](final_assignment/01_0_Directed_Search.ipynb): This file conducts a directed search across policies given a scenario. Three scenarios are defined. The scenario is determined with the case number.

* [02_Directed_Search_Merged_Outputs.ipynb](final_assignment/02_Directed_Search_Merged_Outputs.ipynb): Directed Search outputs from baseline, worstcase, and bestcase merged
In this file we perform a large number of scenarios on the policies obtained from the previous step.

* [03_Directed_Search_Scoring_Filtering.ipynb](final_assignment/03_Directed_Search_Scoring_Filtering.ipynb): Directed Search
This file merges the outputs from the previous baseline, worst case and best case runs and does rudiminary visualisations on them.

* [04_Directed_Search_Scoring.ipynb](final_assignment/04_Directed_Search_Scoring.ipynb): Directed Search
In this final file we calculate the different robustness scores to select a final policy set.
## Output

The outputs from the model can be found in the [directed_search](final_assignment/output/directed_search/), [open_exploration](final_assignment/output/open_exploration/), and [optimization_moro](output/optimization_moro/) folders in the [output](output) directory.

## Data

The input data can be found at [dikeIjssel_alldata.xlsx](final_assignment/data/dikeIjssel_alldata.xlsx) in the [data](data) directory.

## Folders

* [data](final_assignment/data/): Input data for the data cleaning and the model


* [output](final_assignment/output/): Output data from the model for open exploration and directed search


* [preliminary_experiments](final_assignment/preliminary_experiments/): Preliminary experiments including moro optimization 


* [report](final_assignment/report/): Report prepared for the Transport Company including a policy recommendation and political reflection

  