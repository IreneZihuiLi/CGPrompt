

import numpy as np
import pandas as pd


def generate_instruct():
    instruct = "In the domain of Natural Language Processing, I will give you a project title and description, tell me possible concepts that I need to learn to achieve the project. The results should be a list of concepts ONLY: i.e., concept 1; concept 2; concept3,.... "
    instruct += '\n NOTE: no numbers, each concept should be listed and separated by semicolon'

    df = pd.read_csv('../../data_generator/TutorQA/M1_4_suggestion.tsv', sep='\t').values

    instruct_list = []
    for row in df:
        temp = instruct + '\n'
        temp += 'title: {} \n'.format(row[0])
        temp += 'description: {} \n'.format(row[1])
        # temp += 'ONLY OUTPUT CONCEPTS IN LIST AND BE SEPARATED BY ;, NO NUMBERS, BE SHORT ANSWER.'
        # temp += 'SEPARATE BY SEMICOLON ! NO NUMBERS !'


        instruct_list.append(temp)
    np.save(file='./tutor_qa.npy', arr=instruct_list)


if __name__ == "__main__":
    generate_instruct()






