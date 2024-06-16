import json
import os

from tqdm import tqdm
import urllib.request
import json

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
        return "ERROR"

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
                    "max_tokens": 50,
                })

                res = run_request(input_json)
                # print(res)
                res = res.replace('\n','')
                w.write(res+'\n')

if __name__ == "__main__":
# <<<<<<< HEAD
    """ following part for NLP, didn't change (sixun), uncomment & run"""
    # flag = "pos"
    # batch = "1"
    # flag="neg"
    for batch in ['1','2','3','4']:
    # for batch in ['0']:
        for flag in ["neg","pos"]:
            result_path = '../results/1206_train_wiki/t1.'+flag+'.' + batch + '.txt.test'
            instruction_path = '../instruction_test/1206_train_wiki/t1.'+flag+'.' + batch + '.txt'

            generate_txt(result_path,instruction_path)

    """ following part for CV/BIO, , uncomment & run"""
    # domain = 'CV'
    # for batch in range(5):
    #     for flag in ["pos", "neg"]:
    #         # the generated text output_llama path
    #         output_path = '../results/{}'.format(domain)
    #         os.makedirs(output_path, exist_ok=True)
    #
    #         result_path = os.path.join(output_path, 't1.{}.{}.txt.test'.format(flag, batch))
    #         instruction_path = '../instruction_test/{}/t1.{}.{}.txt'.format(domain, flag, batch)
# =======
    # """ following part for NLP, didn't change (sixun), uncomment & run"""
    # # flag = "pos"
    # # batch = "1"
    # # flag="neg"
    # for batch in ['1','2','3','4']:
    # # for batch in ['0']:
    #     for flag in ["neg","pos"]:
    #         result_path = '../results/1205_train/t3.'+flag+'.' + batch + '.txt.test'
    #         instruction_path = '../instruction_test/1205_train/t3.'+flag+'.' + batch + '.txt'
# >>>>>>> 1a772b929ebde7f700c31029dc81cc59db3f1a8d
#     #
#     #         generate_txt(result_path,instruction_path)

#     """ following part for CV/BIO, , uncomment & run"""
#     domain = 'BIO_train'
#     for batch in range(5):
#         for flag in ["pos", "neg"]:
#             # the generated text output_llama path
#             output_path = '../results/{}'.format(domain)
#             os.makedirs(output_path, exist_ok=True)

#             result_path = os.path.join(output_path, 't2.{}.{}.txt.test'.format(flag, batch))
#             instruction_path = '../instruction_test/{}/t2.{}.{}.txt'.format(domain, flag, batch)

#             generate_txt(result_path,instruction_path)

print ('Finished.')
