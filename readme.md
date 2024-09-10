# Graphusion 

Graphusion is a pipeline that extract Knowledge Graph triples from text.

![Architecture](fig_architecture.png)


## Setup
Create a new conda environment and install the required packages:
```
conda create -n graphusion python=3.10
conda activate graphusion
pip install -r requirements.txt
```


## Usage
The pipeline processes text files from the `data/[dataset_name]/raw` directory (e.g., `data/test/raw`) as input. 
Furthermore, the pipeline requires relation definitions as a JSON file. This file defines the relations and 
provides a description of the relation (e.g., `data/test/relation_types.json`). In addition, some information can be 
provided to improve the results (`--gold_concept_file`, `--refined_concepts_file`, `--annotated_graph_file`)
or to skip pipeline steps (`--input_json_file`, `--input_triple_file`). See parameters below.

The ACL data is originally in a csv format. Therefore, we provide the notebook `preprocess.ipynb` to convert the 
data into the required text files.

The pipeline can be run using the following command:

```
usage: main.py [-h] [--run_name RUN_NAME] --dataset DATASET --relation_definitions_file RELATION_DEFINITIONS_FILE [--input_json_file INPUT_JSON_FILE]
               [--input_triple_file INPUT_TRIPLE_FILE] [--model MODEL] [--max_resp_tok MAX_RESP_TOK] [--max_input_char MAX_INPUT_CHAR]
               [--prompt_tpextraction PROMPT_TPEXTRACTION] [--prompt_fusion PROMPT_FUSION] [--gold_concept_file GOLD_CONCEPT_FILE]
               [--refined_concepts_file REFINED_CONCEPTS_FILE] [--annotated_graph_file ANNOTATED_GRAPH_FILE] [--language LANGUAGE] [--verbose]

options:
  -h, --help            show this help message and exit
  --run_name RUN_NAME   Assign a name to this run. The name will be used to, e.g., determine the output directory. We recommend to use unique and descriptive names to
                        distinguish the results of different models.
  --dataset DATASET     Name of the dataset. Is used to, e.g., determine the input directory.
  --relation_definitions_file RELATION_DEFINITIONS_FILE
                        Path to the relation definitions file. The file should be a JSON file, where the keys are the relation types and the values are dictionaries with the
                        following keys: 'label', 'description'.
  --input_json_file INPUT_JSON_FILE
                        Path to the input file. Step 1 will be skipped if this argument is provided. The input file should be a JSON file with the following structure:
                        {'concept1': [{'abstract': ['abstract1', ...], 'label: 0},...} E.g. data/test/concept_abstracts.json is the associated file createddurin step 1 in the
                        test run.
  --input_triple_file INPUT_TRIPLE_FILE
                        Path to the input file storing the triples in the format as outputted by the candidate triple extraction model. Step 1 and step 2 will be skipped if
                        this argument is provided.
  --model MODEL         Name of the LLM that should be used for the KG construction.
  --max_resp_tok MAX_RESP_TOK
                        Maximum number of tokens in the response of the candidate triple extraction model.
  --max_input_char MAX_INPUT_CHAR
                        Maximum number of characters in the input of the candidate triple extraction model.
  --prompt_tpextraction PROMPT_TPEXTRACTION
                        Path to the prompt template for step 1.
  --prompt_fusion PROMPT_FUSION
                        Path to the prompt template for fusion.
  --gold_concept_file GOLD_CONCEPT_FILE
                        Path to a file with concepts that are provided by experts. The file should be a tsv file, each row should look like: 'concept id | concept
  --refined_concepts_file REFINED_CONCEPTS_FILE
                        In step 2 (candidate triple extraction) many new concepts might be added. Instead of using these, concepts can be provided through this parameter.
                        Specify the path to a file with refined concepts of the graph. The file should be a tsv file, each row should look like: "concept id | concept name"
  --annotated_graph_file ANNOTATED_GRAPH_FILE
                        Path to the annotated graph.
  --language LANGUAGE   Language of the abstracts.
  --verbose             Print additional information to the console.
```

The output of the pipeline are the following files: 
- `concept_abstracts`: The json file mapping the extracted concepts to their abstracts.
- `step-02.jsonl`: The extracted triples in linewise JSON format.
- `step-03.jsonl`: The fused triples in linewise JSON format.


## Example 
To run the full pipeline on a small sample (`test`) dataset, call: 
`python main.py --run_name "test" --dataset "test" --relation_definitions_file "data/test/relation_types.json" --gold_concept_file "data/test/gold_concepts.tsv" --refined_concepts_file "data/test/refined_concepts.tsv"`

To reproduce the Graphusion results on the ACL (`nlp) dataset, call:
`python main.py --run_name "acl" --dataset "nlp" --relation_definitions_file "data/nlp/relation_types.json" --gold_concept_file "data/nlp/gold_concepts.tsv" --refined_concepts_file "data/nlp/refined_concepts.tsv"`