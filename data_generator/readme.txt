
This folder contains scripts and sample data for Graph Reasoning. 

----- 2024.1.13 Version M1 -----
Data format is the same.
NEW Data is in `TutorQA` folder.
`output_llama`, `output_gpt3` and `output_gpt4` are preliminary answers from them.

0_load_graph.py: scripts for generating data, each type has 50 question-answer pairs, totally 3 types:
    1. one hop ancestors;
    2. one hop predecessors;
    3. multi hop.

1_llama_test.py: apply llama to generate answers;
2_evaluation.py: simple evaluation, calculating precision, recall and F1. NOTE: this is to be improved on synonyms or abbreviations.

----- 2023.12.26 Version M1 ----- 
Data format:

TSV file, \t separated;
Column 1 is the generated question;
Column 2 is the corresponding concept list, which is seperated by ';' character. 

We use 0_load_graph.py to generated all the data.  
