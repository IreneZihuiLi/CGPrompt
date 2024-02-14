import json
import os
import sys

import numpy as np
from tqdm import tqdm
import urllib.request
import json
from openai import OpenAI

url = "http://bendstar.com:8000/v1/chat/completions"
req_header = {
    'Content-Type': 'application/json',
}

# def run_request(input_json):

#     req = urllib.request.Request(url, data=input_json.encode(), method='POST', headers=req_header)
#     with urllib.request.urlopen(req) as response:
#         body = json.loads(response.read())
#         headers = response.getheaders()
#         status = response.getcode()
#         return body['choices'][0]['message']['content']

def run_request(input_json):
    try:
        req = urllib.request.Request(url, data=input_json.encode(), method='POST', headers=req_header)
        with urllib.request.urlopen(req) as response:
            body = json.loads(response.read())
            # headers = response.getheaders()  # Uncomment if headers are needed
            # status = response.getcode()  # Uncomment if status code is needed
            return body['choices'][0]['message']['content']
    except Exception as e:
        print(e)
        return "ERROR"


def generate_qs(result_path, instruction_path):
    instruct = np.load(instruction_path)

    with open(result_path, 'w') as w:
        for line in tqdm(instruct[:]):
            input_json = json.dumps({
                "model": "meta-llama/Llama-2-70b-chat-hf",
                "messages": [{"role": "system", "content": "You are a Knowledge Graph Builder."},
                             {"role": "user", "content": line}],
                "temperature": 0,
                "max_tokens": 120 #50,
            })

            res = run_request(input_json)

            res = res.replace('\n', ';')
            # res = res.replace('  Sure! Here are some possible concepts you may need to learn to achieve the project:', '')
            res = res.split(':;;')[1]
            res = res.replace('* ', '')

            # print(line)
            # print(res)
            # sys.exit(10)

            w.write(res + '\n')


def generate_qs_gpt(result_path, instruction_path):
    instruct = np.load(instruction_path)

    key = 'sk-' 
    client = OpenAI(api_key=key)

    with open(result_path, 'w') as w:
        for line in tqdm(instruct):
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                # model="gpt-4-1106-preview",
                messages=[
                    {"role": "system",
                     "content": "You are a Knowledge Graph Builder."},
                    {"role": "user", "content": line}
                ],
                max_tokens=100
            )

            res = completion.choices[0].message.content  # string
            res = res.replace('\n', '')

            # print(line)
            # print(res)
            # sys.exit(10)

            w.write(res + '\n')


if __name__ == "__main__":
    generate_qs(result_path='./tutor_qa_llama_70b.txt', instruction_path='./tutor_qa.npy')
    # generate_qs_gpt(result_path='./tutor_qa_gpt35.txt', instruction_path='./tutor_qa.npy')

    print ('Finished.')
