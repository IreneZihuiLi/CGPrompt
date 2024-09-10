import os
import json
import logging
from models import KnowledgeGraphLLM
from argparse import ArgumentParser

from step_01_concept_extraction import step_01_concept_extraction
from step_02_triple_extraction import step_02_triple_extraction
from step_03_fusion import step_03_fusion

if __name__ == "__main__":
    argparse = ArgumentParser()
    argparse.add_argument("--run_name", type=str, default="test",
                          help="Assign a name to this run. The name will be used to, e.g., determine "
                               "the output directory. We recommend to use unique and descriptive names "
                               "to distinguish the results of different models.")
    argparse.add_argument("--dataset", type=str,
                          required=True,
                          help="Name of the dataset. Is used to, e.g., determine the input directory.")
    argparse.add_argument("--relation_definitions_file", type=str,
                          required=True,
                          help="Path to the relation definitions file. The file should be a JSON file, "
                               "where the keys are the relation types and the values are dictionaries "
                               "with the following keys: 'label', 'description'.")

    # these arguments allow to provide the data from the previous steps directly
    # instead of running these steps
    argparse.add_argument("--input_json_file", type=str, default="",
                          help="Path to the input file. Step 1 will be skipped if this argument "
                               "is provided. The input file should be a JSON file with the "
                               "following structure: "
                               "{'concept1': [{'abstract': ['abstract1', ...], 'label: 0},...} "
                               "E.g. data/test/concept_abstracts.json is the associated file created"
                               "durin step 1 in the test run.")
    argparse.add_argument("--input_triple_file", type=str, default="",
                          help="Path to the input file storing the triples in the format as outputted "
                               "by the candidate triple extraction model. Step 1 and step 2 will "
                               "be skipped if this argument is provided.")

    # these arguments allow to configure the LLM model
    argparse.add_argument("--model", type=str, default="gpt-3.5-turbo",
                          help="Name of the LLM that should be used for the KG construction.")
    argparse.add_argument("--max_resp_tok", type=int, default=200,
                          help="Maximum number of tokens in the response of the candidate triple "
                               "extraction model.")
    argparse.add_argument("--max_input_char", type=int, default=10000,
                          help="Maximum number of characters in the input of the candidate triple "
                               "extraction model.")
    argparse.add_argument("--prompt_tpextraction", type=str,
                          default="prompts/prompt_tpextraction.txt",
                          help="Path to the prompt template for step 1.")
    argparse.add_argument("--prompt_fusion", type=str, default="prompts/prompt_fusion.txt",
                          help="Path to the prompt template for fusion.")

    # these arguments allow to provide additional data
    argparse.add_argument("--gold_concept_file", type=str,
                          default="",
                          help="Path to a file with concepts that are provided by experts. "
                               "The file should be a tsv file, each row should look like: "
                               "'concept id | concept")
    argparse.add_argument('--refined_concepts_file', type=str,
                          default="True",
                          help='In step 2 (candidate triple extraction) many new concepts might be '
                               'added. Instead of using these, concepts can be provided through this '
                               'parameter. Specify the path to a file with refined concepts '
                               'of the graph. The file should be a tsv file, each row should look like: '
                               '"concept id | concept name"')
    argparse.add_argument("--annotated_graph_file", type=str,
                          default="data/prerequisite_of_graph.tsv",
                          help="Path to the annotated graph.")

    # language settings
    argparse.add_argument('--language', type=str, default='english',
                          help='Language of the abstracts.')

    # logging
    argparse.add_argument('--verbose', action='store_true',
                          help='Print additional information to the console.')

    # Parse the arguments
    args = argparse.parse_args()
    VERBOSE = args.verbose
    RUN_NAME = args.run_name
    RELATION_DEFINITIONS_FILE = args.relation_definitions_file
    MODEL_NAME = args.model
    MAX_RESPONSE_TOKEN_LENGTH_CANDIDATE_TRIPLE_EXTRACTION = args.max_resp_tok
    PROMPT_TPEXTRACTION_FILE = args.prompt_tpextraction
    PROMPT_FUSION_FILE = args.prompt_fusion

    # --- Setup ---
    # initialize logger
    if VERBOSE:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.INFO
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.info(f"RUN_NAME: {RUN_NAME}")

    # Prepare the output directory
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists(f'output/{RUN_NAME}'):
        os.makedirs(f'output/{RUN_NAME}')

    # write config to output directory
    config = args.__dict__
    json.dump(config, open(f'output/{RUN_NAME}/config.json', 'w'), indent=4)

    # assign output file names if not provided
    if args.input_json_file == "":
        CONCEPT_EXTRACTION_OUTPUT_FILE = f'output/{RUN_NAME}/concepts.tsv'
        CONCEPT_ABSTRACTS_OUTPUT_FILE = f'output/{RUN_NAME}/concept_abstracts.json'
    else:
        CONCEPT_ABSTRACTS_OUTPUT_FILE = args.input_json_file
        logging.info(
            f"Using provided input file: {CONCEPT_ABSTRACTS_OUTPUT_FILE} "
            f"(skipping step 1 - concept extraction).")

    if args.input_triple_file == "":
        TRIPLE_EXTRACTION_OUTPUT_FILE = f'output/{RUN_NAME}/step-02.jsonl'
    else:
        TRIPLE_EXTRACTION_OUTPUT_FILE = args.input_triple_file
        logging.info(
            f"Using provided input file: {TRIPLE_EXTRACTION_OUTPUT_FILE} "
            f"(skipping step 2 - triple extraction).")

    # output file of the pipeline
    FUSION_OUTPUT_FILE = f'output/{RUN_NAME}/step-03.jsonl'

    # Load the relation definitions
    relation_def = json.load(open(RELATION_DEFINITIONS_FILE, 'r'))
    relation_types = list(relation_def.keys())
    relation_2_id = {v: k for k, v in enumerate(relation_types)}
    id_2_relation = {k: v for k, v in enumerate(relation_types)}

    # Configure API keys
    os.environ["OPENAI_API_KEY"] = json.load(open('private_config.json'))['OPENAI_API_KEY']

    # init the LLM
    model = KnowledgeGraphLLM(model_name=MODEL_NAME,
                              max_tokens=MAX_RESPONSE_TOKEN_LENGTH_CANDIDATE_TRIPLE_EXTRACTION)

    # --- Pipeline ---
    if args.input_json_file == "" and args.input_triple_file == "":
        # load raw text data
        if len(os.listdir(f'data/{args.dataset}/raw/')) == 0:
            logging.error(f"No input text files found in data/{args.dataset}/raw/.")
        texts = []
        for file in os.listdir(f'data/{args.dataset}/raw/'):
            if file.endswith('.txt'):
                logging.info(f"Loading file: {file}")
                with open(f'data/{args.dataset}/raw/{file}', 'r') as f:
                    for line in f:
                        texts.append(line)

        # extract concepts
        step_01_concept_extraction(texts=texts,
                                   concept_extraction_output_file=CONCEPT_EXTRACTION_OUTPUT_FILE,
                                   concept_abstracts_output_file=CONCEPT_ABSTRACTS_OUTPUT_FILE,
                                   logging=logging,
                                   config=config)

    # Load the abstract data (either created in step 1 or provided as input)
    data = json.load(open(CONCEPT_ABSTRACTS_OUTPUT_FILE, 'r'))

    if args.input_triple_file == "":
        step_02_triple_extraction(model=model,
                                  output_file=TRIPLE_EXTRACTION_OUTPUT_FILE,
                                  relation_def=relation_def,
                                  data=data,
                                  logging=logging,
                                  config=config)

    step_03_fusion(model=model,
                   input_file=TRIPLE_EXTRACTION_OUTPUT_FILE,
                   output_file=FUSION_OUTPUT_FILE,
                   relation_def=relation_def,
                   relation_2_id=relation_2_id,
                   data=data,
                   logging=logging,
                   config=config)
