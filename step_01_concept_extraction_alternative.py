import json
import os
import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm


# TODO Experimental: Not yet fully implemented and tested


def step_00(model: any,
            raw_text_dir: str,
            output_file: str,
            relation_def: dict[str, dict[str, str]],
            logging: any,
            config: dict[str, any]):
    """
    Extract candidate concepts from the data and write them to the output file.
    """
    #logging.info("Step 0: Starting candidate concept extraction.")
    # iterate over files in raw_text_dir

    # todo these will be function parameters
    num_samples = 1000
    model = "gpt-3.5-turbo"
    max_response_tokens = 200
    os.environ["OPENAI_API_KEY"] = json.load(open('private_config.json'))['OPENAI_API_KEY']
    model = ChatOpenAI(model=model, max_tokens=max_response_tokens)

    candidate_concepts = set()
    input_texts = json.load(open('data/japanese_med/raw.json'))['text']

    # Ensure there are enough elements in the list to sample from
    if len(input_texts) < num_samples:
        print(f"The list contains fewer than {num_samples} abstracts.")
    else:
        # Get random samples from the list
        input_texts = random.sample(input_texts, num_samples)

    prompt_template_txt = open(config['prompt_step_00']).read()
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge graph builder."),
        ("user", prompt_template_txt)
    ])

    for text in tqdm(input_texts):
        p = prompt_template.invoke(
            {"abstracts": text[:config['max_input_char']],
             "relation_definitions": '\n'.join(
                 [f"{rel_type}: {rel_data['description']}" for rel_type, rel_data in
                  relation_def.items()])}
        )

        # query the model
        o = model.invoke(p)
        concepts = [x.strip().lower() for x in o.content.split(',')]
        candidate_concepts = candidate_concepts.union(set(concepts))

    with open(output_file, 'w') as f:
        for concept in candidate_concepts:
            f.write(concept + '\n')


if __name__ == '__main__':
    config = {'prompt_step_00': 'prompts/prompt_step_00.txt',
              'max_input_char': 10000}
    step_00(None, 'data/japanese_med/raw', 'output/japanese_med/concepts.tsv', {}, {}, config)





