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

one_shot = '''
ONLY answer with topics (keywords or phrases) with out any explanation. Here is an example: 
<Question>: In the domain of natural language processing, I want to learn about optimization, what concepts should I learn frist? </Question>	
<Answer>: 1.probabilities; 2. calculus; 3.linear algebra;</Answer> \n
'''
def generate_txt_gpt(result_path,instruction_path):
    key = 'YOURKEY'
    client = OpenAI(api_key=key)

    with open(result_path,'w') as w:
        with open(instruction_path,'r') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc="Processing"):
                question, answer = line.split("?")
                question = one_shot + '<Question>:' + question
                question += "?</Question> <Answer>: "
                completion = client.chat.completions.create(
                    # model="gpt-3.5-turbo",
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system",
                         "content": "You are an natural language processing tutor, answer questions. "},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=100
                )

                res = completion.choices[0].message.content  # string
                # res = res.replace('\n','')
                # w.write(res+'\n')

                res = res.replace('\n', '')
                res = res + " ***** " + answer
                w.write(res + '\n')
                # import pdb;pdb.set_trace()
                # pass

def generate_txt_question5(result_path,instruction_path):
    key = 'sk-'
    client = OpenAI(api_key=key)

    with open(result_path,'w') as w:
        with open(instruction_path,'r') as f:
            lines = f.readlines()[1:]
            for line in tqdm(lines, desc="Processing"):
                concepts, question = line.split("\t")

                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    # model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system",
                         "content": "You are an natural language processing tutor, answer questions. "},
                        {"role": "user", "content": question}
                    ],
                    max_tokens=200
                )

                res = completion.choices[0].message.content  # string
                # res = res.replace('\n','')
                # w.write(res+'\n')

                res = res.replace('\n', '')

                w.write(res + '\n')
                # import pdb;pdb.set_trace()
                # pass

if __name__ == "__main__":


    # # fname = 'M1_3_multi_hop.tsv'
    # fname = 'M1_2_one_hop.tsv'
    # # fname = 'M1_one_hop_pds.tsv'
    # instruction_path = 'TutorQA_test/' + fname
    # result_path = 'output_gpt4/' + fname
    #
    # generate_txt_gpt(result_path, instruction_path)

    instruction_path = 'TutorQA/M1_5_idea.tsv'
    result_path = 'output_gpt3/M1_5_idea.tsv'
    generate_txt_question5(result_path, instruction_path)

print ('Finished.')
