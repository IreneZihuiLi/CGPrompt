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


def concept_run():
    DOMAIN = 'BIO'
    concept_path = '../concept_data/BIO_topics.tsv'
    concept_generated_text = []
    concept_generated_emb = []

    concept_data = dict()
    with open(concept_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('|')
            concept_data[int(key)-1] = value.strip()
    print (len(concept_data)," concepts loaded.")

    for key,value in concept_data.items():
        print(f"c{key}: ", value)

        messages = [
            {
                "role": "system",
                "content": f"You are a {DOMAIN} expert.",
            },
            {
                "role": "user",
                "content": f"What's the definetion of '{value}'"},
        ]

        text, emb = run_and_store(messages)
        concept_generated_text.append(text)
        concept_generated_emb.append(emb)

    with open(f'./{DOMAIN}_concept_generated_text.pkl', 'wb') as f:
        pickle.dump(concept_generated_text, f)

    with open(f'./{DOMAIN}_concept_generated_emb.pkl', 'wb') as f:
        pickle.dump(concept_generated_emb, f)


if __name__ == '__main__':
    concept_run()