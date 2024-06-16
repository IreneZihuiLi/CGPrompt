import pickle
import torch
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer

doc_max_token_len = 4000
llm = LLM(model='meta-llama/Llama-2-70b-chat-hf',
          tensor_parallel_size=4, trust_remote_code=True,
          enforce_eager=True)

engine_model_config = llm.llm_engine.get_model_config()
max_model_len = engine_model_config.max_model_len
tokenizer = get_tokenizer(
        engine_model_config.tokenizer,
        tokenizer_mode=engine_model_config.tokenizer_mode,
        trust_remote_code=engine_model_config.trust_remote_code)


def run_and_store(input_message):
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=200)

    prompt = tokenizer.apply_chat_template(
        conversation=input_message,
        tokenize=False,
        add_generation_prompt=True)
                        
    output, hidden = llm.generate(prompt, sampling_params)
    print("hidden shape: ", hidden.shape)

    generated_text = output[0].outputs[0].text
    print(f"Prompt: {prompt!r} \nGenerated text: {generated_text!r}")

    return generated_text, hidden.view(-1)



with open('save/graph.pkl', 'rb') as f:
    G = pickle.load(f)

def concept_run():
    concept_generated_text = []
    concept_generated_emb = []

    for concept_id in range(1, 322 + 1):
        concept_text = G.nodes[f'c{concept_id}']['text']
        print(f"c{concept_id}: ", concept_text)
            
        messages = [
            {
                "role": "system",
                "content": "You are a NLP expert.",
            },
            {
                "role": "user", 
                "content": f"What's the definetion of '{concept_text}'"},
        ]

        text, emb = run_and_store(messages)
        concept_generated_text.append(text)
        concept_generated_emb.append(emb)
        
    with open('./concept_generated_text.pkl', 'wb') as f:
        pickle.dump(concept_generated_text, f)

    with open('./concept_generated_emb.pkl', 'wb') as f:
        pickle.dump(concept_generated_emb, f)


def document_run():
    doc_generated_text = []
    doc_generated_emb = []

    for doc_id in range(1, 1836 + 1):
        print(f"d{doc_id}")
        tokens = tokenizer.encode(G.nodes[f"d{doc_id}"]['text'], add_special_tokens=False)
        first_n_tokens = tokens[:doc_max_token_len]  # n is the number of tokens you want to keep
        new_doc_content = tokenizer.decode(first_n_tokens)

        neigh_concept_list = []
        for neigh_concept in G.neighbors(f'd{doc_id}'):
            neigh_concept_list.append(G.nodes[neigh_concept]['text'])
        print("len of list: ", len(neigh_concept_list))

        messages = [
            {
                "role": "system",
                "content": "You are a NLP expert.",
            },
            {
                "role": "user", 
                "content": f"""
                ## Concepts
                {neigh_concept_list}

                ## Document
                {new_doc_content}

                ## Task
                Please summary the content of relevant concepts from the text in the concepts list. If the concepts list is empty, please summarize the content of the text.
                """},
        ]
        text, emb = run_and_store(messages)
        doc_generated_emb.append(emb)
        doc_generated_text.append(text)


    with open('./doc_generated_text.pkl', 'wb') as f:
        pickle.dump(doc_generated_text, f)

    with open('./doc_generated_emb.pkl', 'wb') as f:
        pickle.dump(doc_generated_emb, f)


if __name__ == '__main__':
    document_run()
    concept_run()