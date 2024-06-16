import torch
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm import LLM, SamplingParams

# Specific Version
# git clone https://github.com/tinyrolls/vllm
# pip install -e .

llm = LLM(model='meta-llama/Llama-2-70b-chat-hf', tensor_parallel_size=4, trust_remote_code=True)

engine_model_config = llm.llm_engine.get_model_config()
tokenizer = get_tokenizer(
        engine_model_config.tokenizer,
        tokenizer_mode=engine_model_config.tokenizer_mode,
        trust_remote_code=engine_model_config.trust_remote_code)

messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Tell me a story about flower"
    }]


prompt = tokenizer.apply_chat_template(
    conversation=messages,
    tokenize=False,
    add_generation_prompt=True)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=200)

                        
output, hidden = llm.generate(prompt, sampling_params)
print("hidden shape: ", hidden.shape)

generated_text = output[0].outputs[0].text
print(f"Prompt: {prompt!r} \nGenerated text: {generated_text!r}")