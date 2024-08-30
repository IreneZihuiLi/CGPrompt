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


def step_01():
    # --- Step 1: Candidate Triple Extraction ---
    output_stream = open(STEP01_OUTPUT_FILE, 'w')

    # load model
    model = KnowledgeGraphLLM(model_name=MODEL_NAME,
                              max_tokens=MAX_RESPONSE_TOKEN_LENGTH_CANDIDATE_TRIPLE_EXTRACTION)

    # initialize the prompt template
    prompt_template_txt = open(PROMPT_STEP_01_FILE).read()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge graph builder."),
        ("user", prompt_template_txt)
    ])

    # iterate over the data, extract triples and write them to the output stream
    extracted_relations = []
    for concept_id, (concept_name, concept_data) in enumerate(data.items()):
        abstracts = ' '.join(data[concept_name]['abstracts'])

        # instantiate the prompt template
        prompt = prompt_template.invoke(
            {"abstracts": abstracts[:MAX_INPUT_CHAR_LENGTH_ABSTRACTS_CANDIDATE_TRIPLE_EXTRACTION],
             "concepts": [concept_name],
             "relation_definitions": '\n'.join(
                 [f"{rel_type}: {rel_data['description']}" for rel_type, rel_data in
                  relation_def.items()])})

        # query the model
        response = model.invoke(prompt)

        if response != "No triples extracted.":
            response_json = json.loads(response)
            for triple in response_json:
                if triple['p'] not in relation_types:
                    continue
                else:
                    extracted_relations.append(triple['p'])

                triple['id'] = concept_id
                triple['concept'] = concept_name
                output_stream.write(json.dumps(triple) + '\n')

    output_stream.close()

    logging.info("Step 1: Candidate Triple Extraction completed.")
    logging.info(f"Num extracted candidate triples: {len(extracted_relations)}")
    logging.debug(f"Extracted candidate triples: {Counter(extracted_relations)}")


def step_fusion():
    # --- Step 2: Fusion ---
    candidate_triples = []
    for line in open(STEP01_OUTPUT_FILE, 'r'):
        t = json.loads(line)
        candidate_triples.append((t['s'], t['p'], t['o']))

    if REFINED_CONCEPTS_FILE is not None:
        logging.info(f"Refined concepts specified. Loading concepts from {REFINED_CONCEPTS_FILE}.")
        id_2_concept = {i: str(c['concept']) for i, c in
                        pd.read_csv('data/refined_concepts.tsv', sep='|', header=None,
                                    names=['id', 'concept'], index_col=0).iterrows()}
        logging.info(
            f"Loaded {len(id_2_concept)} refined concepts, e.g. {', '.join(list(id_2_concept.values())[:3])}")
    else:
        # randomly sample up to 100 concepts from step 1
        concepts = [c[0] for c in candidate_triples] + [c[2] for c in candidate_triples]
        random.shuffle(concepts)
        logging.info(
            f'No refined concepts specified. Randomly selected concepts: {", ".join(concepts[:100])}')
        id_2_concept = {i: c for i, c in enumerate(concepts)}

    concept_2_id = {v: k for k, v in id_2_concept.items()}

    # build the prerequisite-of graph
    prerequisite_of_triples = []
    with open('data/prerequisite-of_graph.tsv', 'r') as f:
        for line in f:
            s, p, o = line.strip().split('\t')
            prerequisite_of_triples.append((str(s), str(p), str(o)))

    prerequisite_of_graph = get_nx_graph(prerequisite_of_triples, concept_2_id, relation_2_id)

    # todo refactor that the model is just loaded once for step01 and fusion
    # load model
    model = KnowledgeGraphLLM(model_name=MODEL_NAME,
                              max_tokens=MAX_RESPONSE_TOKEN_LENGTH_CANDIDATE_TRIPLE_EXTRACTION)

    # initialize the prompt template
    prompt_template_txt = open(PROMPT_FUSION_FILE).read()

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge graph builder."),
        ("user", prompt_template_txt)
    ])

    output_stream = open(FUSION_OUTPUT_FILE, 'w')
    for id, candidate_concept in id_2_concept.items():
        candidate_subgraph = verbalize_neighbors_triples_from_triples(candidate_triples,
                                                                      candidate_concept)
        if len(candidate_subgraph) <= 3 and data.keys() >= 3:
            continue
        # todo why are only outgoing hypernym edges used?
        prerequisite_of_graph_subgraph = verbalize_neighbors_triples_from_graph(
            prerequisite_of_graph, candidate_concept, concept_2_id, id_2_concept, mode='outgoing')
        abstracts = ' '.join(
            data[candidate_concept]['abstracts']) if candidate_concept in data else ''

        prompt = prompt_template.invoke(
            {"concept": candidate_concept,
             "graph1": candidate_subgraph,
             "graph2": prerequisite_of_graph_subgraph,
             "background": abstracts,
             "relation_definitions": '\n'.join(
                 [f"{rel_type}: {rel_data['description']}" for rel_type, rel_data in
                  relation_def.items()])})

        # query the model
        response = model.invoke(prompt)

        if response != "No triples extracted.":
            response_json = json.loads(response)
            for triple in response_json:
                if triple['p'] not in relation_types:
                    continue
                output_stream.write(json.dumps(triple) + '\n')
    output_stream.close()
    logging.info("Fusion: Completed.")


# todo implement hydra for configuration management
if __name__ == "__main__":
    argparse = ArgumentParser()
    # todo complete the argument list
    argparse.add_argument("--run_name", type=str, default="test",
                          help="Name of the run. Is used to, e.g., determine the output directory.")
    argparse.add_argument("--input_file", type=str, default="data/concept_abstracts_sample.json",
                          help="Path to the input file. The input file should be a JSON file with the "
                               "following structure: "
                               "{'concept1': [{'abstract': ['abstract1', ...], 'label: 0},...} ")
    argparse.add_argument("--model", type=str, default="gpt-3.5-turbo",
                          help="Name of the LLM.")
    argparse.add_argument("--max_resp_tok", type=int, default=40,
                          help="Maximum number of tokens in the response of the candidate triple "
                               "extraction model.")
    argparse.add_argument("--max_input_char", type=int, default=14500,
                          help="Maximum number of characters in the input of the candidate triple "
                               "extraction model.")
    argparse.add_argument("--prompt_step_01", type=str, default="prompts/prompt_step_01.txt",
                          help="Path to the prompt template for step 1.")
    argparse.add_argument("--prompt_fusion", type=str, default="prompts/prompt_fusion.txt",
                          help="Path to the prompt template for fusion.")
    argparse.add_argument('--verbose', action='store_true',
                          help='Print additional information to the console.')
    argparse.add_argument('--refined_concepts_file', type=str, default=None)

    # Parse the arguments
    args = argparse.parse_args()
    RUN_NAME = args.run_name
    CONCEPTS_ABSTRACTS_FILE = args.input_file
    MODEL_NAME = args.model
    MAX_RESPONSE_TOKEN_LENGTH_CANDIDATE_TRIPLE_EXTRACTION = args.max_resp_tok
    MAX_INPUT_CHAR_LENGTH_ABSTRACTS_CANDIDATE_TRIPLE_EXTRACTION = args.max_input_char
    PROMPT_STEP_01_FILE = args.prompt_step_01
    PROMPT_FUSION_FILE = args.prompt_fusion
    VERBOSE = args.verbose
    REFINED_CONCEPTS_FILE = args.refined_concepts_file
    # todo write config to output directory
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
    relation_def = json.load(open('data/relation_types_new.json'))
    relation_types = list(relation_def.keys())
    relation_2_id = {v: k for k, v in enumerate(relation_types)}
    id_2_relation = {k: v for k, v in enumerate(relation_types)}

    # Prepare the output directory
    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists(f'output/{RUN_NAME}'):
        os.makedirs(f'output/{RUN_NAME}')

    # Configure API keys
    os.environ["OPENAI_API_KEY"] = json.load(open('private_config.json'))['OPENAI_API_KEY']

    # todo uncomment after debugging
    # step_01()
    # step_fusion()


