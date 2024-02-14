import json
import os

from tqdm import tqdm
import urllib.request
import json

url = "http://bendstar.com:8000/v1/chat/completions"
req_header = {
    'Content-Type': 'application/json',
}

def run_request(input_json):
    try:
        req = urllib.request.Request(url, data=input_json.encode(), method='POST', headers=req_header)
        with urllib.request.urlopen(req) as response:
            body = json.loads(response.read())
            # headers = response.getheaders()  # Uncomment if headers are needed
            # status = response.getcode()  # Uncomment if status code is needed
            return body['choices'][0]['message']['content']
    except Exception as e:
        return "ERROR"


one_shot = '''
ONLY answer with topics (keywords or phrases) with out any explanation. Here is an example: 
<Question>: In the domain of natural language processing, I want to learn about optimization, what concepts should I learn frist? </Question>	
<Answer>: 1.probabilities; 2. calculus; 3.linear algebra;</Answer> \n
'''


def generate_txt(result_path,instruction_path):
    with open(result_path,'w') as w:
        with open(instruction_path,'r') as f:
            lines = f.readlines()

            for line in tqdm(lines, desc="Processing"):
                question, answer = line.split("?")
                question = one_shot + '<Question>:' +question
                question += "?</Question> <Answer>: "
                # 'truncate lines after question mark'
                # print (question)
                input_json = json.dumps({
                    "model": "meta-llama/Llama-2-70b-chat-hf",
                    "messages": [{"role": "system", "content": "You are an natural language processing tutor, answer questions. "},
                                 {"role": "user",   "content": question}],
                    "temperature": 0,
                    "max_tokens": 200,
                })

                res = run_request(input_json)
                # print(res)
                res = res.replace('\n','\t')
                res = res +" ***** "+answer
                w.write(res)

def generate_txt_question5(result_path,instruction_path):
    with open(result_path,'w') as w:
        with open(instruction_path,'r') as f:
            lines = f.readlines()[1:]

            for line in tqdm(lines, desc="Processing"):
                answer, question = line.split("\t")

                input_json = json.dumps({
                    "model": "meta-llama/Llama-2-70b-chat-hf",
                    "messages": [{"role": "system", "content": "You are an natural language processing tutor, answer questions. "},
                                 {"role": "user",   "content": question+' Show one possible project only.'}],
                    "temperature": 0,
                    "max_tokens": 250,
                })

                res = run_request(input_json)
                # print(res)
                res = res.replace('\n',' ')
                # res = answer +'\t' +res
                w.write(res+'\n')
                # import pdb;pdb.set_trace()
                pass

if __name__ == "__main__":


    # # fname = 'M1_3_multi_hop.tsv'
    # fname = 'M1_2_one_hop.tsv'
    # # fname = 'M1_one_hop_pds.tsv'
    # instruction_path= 'TutorQA_test/'+fname
    # result_path= 'output_llama/'+fname
    #
    # generate_txt(result_path,instruction_path)

    instruction_path = 'TutorQA/M1_5_idea.tsv'
    result_path = 'output_llama/M1_5_idea.tsv'
    generate_txt_question5(result_path, instruction_path)
    
print ('Finished.')
