import os
import json
from tqdm import tqdm
import urllib.request
import json
from openai import OpenAI

url = "http://bendstar.com:8000/v1/chat/completions"
req_header = {
    'Content-Type': 'application/json',
}

def run_request(input_json):

    req = urllib.request.Request(url, data=input_json.encode(), method='POST', headers=req_header)
    with urllib.request.urlopen(req) as response:
        body = json.loads(response.read())
        headers = response.getheaders()
        status = response.getcode()
        return body['choices'][0]['message']['content']

def generate_txt(result_path,instruction_path):
    with open(result_path,'w') as w:
        with open(instruction_path,'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Processing"):

                input_json = json.dumps({
                    "model": "meta-llama/Llama-2-70b-chat-hf",
                    "messages": [{"role": "system", "content": "You are a Knowledge Graph Builder."},
                                 {"role": "user",   "content": line}],
                    "temperature": 0,
                })

                res = run_request(input_json)
                # print(res)
                res = res.replace('\n','')
                w.write(res+'\n')

def generate_txt_gpt(result_path,instruction_path):
    # client = OpenAI()
    key = 'sk-' 
    client = OpenAI(api_key=key)

    with open(result_path,'w') as w:
        with open(instruction_path,'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Processing"):
                completion = client.chat.completions.create(
                    # model="gpt-3.5-turbo",
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system",
                         "content": "You are a Knowledge Graph Builder."},
                        {"role": "user", "content": line+' Then explain your reason.'}
                    ],
                    max_tokens=100
                )

                res = completion.choices[0].message.content  # string
                res = res.replace('\n','')
                w.write(res+'\n')
                # import pdb;pdb.set_trace()
                # pass

if __name__ == "__main__":


    """ following part for CV/BIO, , uncomment & run"""
    domain = 'BIO' #'CV'
    for batch in range(5):
        for flag in ["pos", "neg"]:
            # the generated text output_llama path
            # output_path = '../results/{}_gpt35'.format(domain)
            output_path = '../results/{}_gpt4'.format(domain)

            os.makedirs(output_path, exist_ok=True)

            result_path = os.path.join(output_path, 't1.{}.{}.txt.test'.format(flag, batch))
            instruction_path = '../instruction_test/{}/t1.{}.{}.txt'.format(domain, flag, batch)

            generate_txt_gpt(result_path, instruction_path)

print ('Finished.')
