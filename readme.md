# Graphusion 

## Setup
Create a new conda environment and install the required packages:
```
conda create -n graphusion python=3.10
conda activate graphusion
pip install -r requirements.txt
```


## Usage
The pipeline can be run using the following command:

# todo Update when final version is ready
```
main.py [-h] [--run_name RUN_NAME] [--input_file INPUT_FILE] [--model MODEL] [--max_resp_tok MAX_RESP_TOK] [--max_input_char MAX_INPUT_CHAR]
               [--prompt_step_01 PROMPT_STEP_01] [--prompt_fusion PROMPT_FUSION] [--verbose] [--refined_concepts_file REFINED_CONCEPTS_FILE]

options:
  -h, --help            show this help message and exit
  --run_name RUN_NAME   Name of the run. Is used to, e.g., determine the output directory.
  --input_file INPUT_FILE
                        Path to the input file. The input file should be a JSON file with the following structure: {'concept1': [{'abstract': ['abstract1', ...], 'label:
                        0},...}
  --model MODEL         Name of the LLM.
  --max_resp_tok MAX_RESP_TOK
                        Maximum number of tokens in the response of the candidate triple extraction model.
  --max_input_char MAX_INPUT_CHAR
                        Maximum number of characters in the input of the candidate triple extraction model.
  --prompt_step_01 PROMPT_STEP_01
                        Path to the prompt template for step 1.
  --prompt_fusion PROMPT_FUSION
                        Path to the prompt template for fusion.
  --verbose             Print additional information to the console.
  --refined_concepts_file REFINED_CONCEPTS_FILE
```


## Example 
To run the pipeline on a very small sample dataset, simply call: `python main.py`

