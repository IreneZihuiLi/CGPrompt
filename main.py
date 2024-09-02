import pandas as pd
import os
import json
import logging
import random
from langchain_core.prompts import ChatPromptTemplate

from graphs import get_nx_graph, TRIPLE_VERB_TEMPLATE, verbalize_neighbors_triples_from_triples, \
    verbalize_neighbors_triples_from_graph
from models import KnowledgeGraphLLM
from argparse import ArgumentParser
from collections import Counter

from step_01 import step_01_new
from step_fusion import step_fusion

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--run_name", type=str, default="test",
                          help="Name of the run. Is used to, e.g., determine the output directory.")
    argparse.add_argument("--input_file", type=str, default="data/concept_abstracts_sample.json",
                          help="Path to the input file. The input file should be a JSON file with the "
                               "following structure: "
                               "{'concept1': [{'abstract': ['abstract1', ...], 'label: 0},...} ")
    argparse.add_argument("--relation_definitions_file", type=str, default="data/relation_types.json",
                          help="Path to the relation definitions file. The file should be a JSON file, "
                               "where the keys are the relation types and the values are dictionaries "
                               "with the following keys: 'label', 'description'.")
    argparse.add_argument("--model", type=str, default="nlp_gpt-3.5-turbo",
                          help="Name of the LLM that should be used for the KG construction.")
    argparse.add_argument("--max_resp_tok", type=int, default=200,
                          help="Maximum number of tokens in the response of the candidate triple "
                               "extraction model.")
    argparse.add_argument("--max_input_char", type=int, default=10000,
                          help="Maximum number of characters in the input of the candidate triple "
                               "extraction model.")
    argparse.add_argument("--prompt_step_01", type=str, default="prompts/prompt_step_01.txt",
                          help="Path to the prompt template for step 1.")
    argparse.add_argument("--annotated_graph_file", type=str, default="data/prerequisite_of_graph.tsv",
                          help="Path to the annotated graph.")
    argparse.add_argument("--prompt_fusion", type=str, default="prompts/prompt_fusion.txt",
                          help="Path to the prompt template for fusion.")
    argparse.add_argument('--verbose', action='store_true',
                          help='Print additional information to the console.')
    argparse.add_argument('--refined_concepts_file', type=str, default='data/refined_concepts.tsv',
                          help='Path to a file with refined concepts of the graph. '
                               'The file should be a tsv file, each row should look like: '
                               '"concept id | concept name"')

    # Parse the arguments
    args = argparse.parse_args()
    RUN_NAME = args.run_name
    CONCEPTS_ABSTRACTS_FILE = args.input_file
    MODEL_NAME = args.model
    MAX_RESPONSE_TOKEN_LENGTH_CANDIDATE_TRIPLE_EXTRACTION = args.max_resp_tok
    PROMPT_STEP_01_FILE = args.prompt_step_01
    PROMPT_FUSION_FILE = args.prompt_fusion
    VERBOSE = args.verbose
    RELATION_DEFINITIONS_FILE = args.relation_definitions_file

    # define output files
    STEP01_OUTPUT_FILE = f'output/{RUN_NAME}/step-01.jsonl'
    FUSION_OUTPUT_FILE = f'output/{RUN_NAME}/fusion.jsonl'

    # initialize logger
    if VERBOSE:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info(f"RUN_NAME: {RUN_NAME}")

    # Load the data
    data = json.load(open(CONCEPTS_ABSTRACTS_FILE, 'r'))


    relation_def = json.load(open(RELATION_DEFINITIONS_FILE, 'r'))



    relation_types = list(relation_def.keys())
    relation_2_id = {v: k for k, v in enumerate(relation_types)}
    id_2_relation = {k: v for k, v in enumerate(relation_types)}

    # Prepare the output directory
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists(f'output/{RUN_NAME}'):
        os.makedirs(f'output/{RUN_NAME}')

    # write config to output directory
    config = args.__dict__
    json.dump(config, open(f'output/{RUN_NAME}/config.json', 'w'), indent=4)

    # Configure API keys
    os.environ["OPENAI_API_KEY"] = json.load(open('private_config.json'))['OPENAI_API_KEY']

    # init the model
    model = KnowledgeGraphLLM(model_name=MODEL_NAME,
                              max_tokens=MAX_RESPONSE_TOKEN_LENGTH_CANDIDATE_TRIPLE_EXTRACTION)

    # todo uncomment after debugging
    #step_01_new(model=model,
    #            output_file=STEP01_OUTPUT_FILE,
    #            relation_def=relation_def,
    #            data=data,
    #            logging=logging,
    #            config=config)

    step_fusion(model=model,
                input_file=STEP01_OUTPUT_FILE,
                output_file=FUSION_OUTPUT_FILE,
                relation_def=relation_def,
                relation_2_id=relation_2_id,
                data=data,
                logging=logging,
                config=config)





