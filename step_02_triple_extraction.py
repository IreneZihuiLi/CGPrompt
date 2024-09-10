import json
from langchain_core.prompts import ChatPromptTemplate
from collections import Counter

from tqdm import tqdm


def step_02_triple_extraction(model: any,
                              output_file: str,
                              relation_def: dict[str, dict[str, str]],
                              data: dict[str, dict[str, list[str]]],
                              logging: any,
                              config: dict[str, any]):
    """
    Step 1: Candidate Triple Extraction
    Extracts candidate triples from the data and writes them to the output file.

    :param model: the language model to use
    :param output_file: the file to write the extracted triples to
    :param relation_def: the relation definitions
    :param data: the data to extract triples from
    :param logging: the logger
    :param config: the configuration can be provided with the following keys: prompt_tpextraction,
    max_input_char

    :return: None
    """

    if 'prompt_tpextraction' not in config:
        config['prompt_tpextraction'] = "prompts/prompt_tpextraction.txt"
        logging.info(f"No prompt template for triple extraction provided. "
                     f"Using default prompt: {config['prompt_tpextraction']}")
    if 'max_input_char' not in config:
        config['max_input_char'] = 10000
        logging.info(f"No max_input_char provided. Using default value: {config['max_input_char']}")


    logging.info("Step 1: Starting candidate triple extraction.")
    output_stream = open(output_file, 'w')

    # initialize the prompt template
    prompt_template_txt = open(config['prompt_tpextraction']).read()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge graph builder."),
        ("user", prompt_template_txt)
    ])

    # iterate over the data, extract triples and write them to the output stream
    extracted_relations = []
    for concept_id, (concept_name, concept_data) in tqdm(enumerate(data.items()), total=len(data)):
        abstracts = ' '.join(data[concept_name]['abstracts'])

        # instantiate the prompt template
        prompt = prompt_template.invoke(
            {"abstracts": abstracts[:config['max_input_char']],
             "concepts": [concept_name],
             "relation_definitions": '\n'.join(
                 [f"{rel_type}: {rel_data['description']}" for rel_type, rel_data in
                  relation_def.items()])})

        # query the model
        response = model.invoke(prompt)

        if response != "None":
            response_json = json.loads(response)
            for triple in response_json:
                if triple['p'] not in list(relation_def.keys()):
                    continue
                else:
                    extracted_relations.append(triple['p'])

                triple['id'] = concept_id
                triple['concept'] = concept_name
                output_stream.write(json.dumps(triple) + '\n')

    output_stream.close()

    logging.info("Step 1: Candidate Triple Extraction completed.")
    logging.info(f"Num extracted candidate triples: {len(extracted_relations)}")
    logging.debug(f"Extracted candidate triples by relaton type: {Counter(extracted_relations)}")
