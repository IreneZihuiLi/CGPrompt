import re

def count_responses(filename):
    with open(filename, 'r') as file:
        content = file.read()
    
    responses = content.split('\n ---------------- \n')

    count_yes, count_no, count_confused, count_error = 0, 0, 0, 0

    for response in responses:
        contains_yes = bool(re.search(r'\b(yes|YES|Yes)\b', response))
        contains_no = bool(re.search(r'\b(no|NO|No)\b', response))

        if contains_yes and not contains_no:
            count_yes += 1
        elif contains_no and not contains_yes:
            count_no += 1
        elif contains_yes and contains_no:
            count_confused += 1
            print("Confused response: ", response)
        else:
            count_error += 1
            print("Error response: ", response)

    return count_yes, count_no, count_confused, count_error

if __name__ == "__main__":
    flag = "pos"
    batch = "0"
    # flag="neg"

    for flag in ["pos","neg"]:
    # for flag in ["pos"]:
        print(f"Processing {flag} batch {batch}...")
        RESULT_PATH = '../results/1116/t1.'+flag+'.' + batch + '.txt.test'
        yes, no, confused, error = count_responses(RESULT_PATH)
        print(f"Yes: {yes}, No: {no}, Confused: {confused}, Error: {error}")
        print("-------------------------------------")
