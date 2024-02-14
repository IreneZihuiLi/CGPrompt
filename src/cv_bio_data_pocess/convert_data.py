__date__ = '2023-11-18'
__author__ = 'Sixun Ouyang'
__version__ = '0.1'
__description__ = 'This file works for CV and BIO data' \
                   'we convert topic and annotation data to match the inputs of LLM in src folder' \
                   'for topic data, save to concept_data folder' \
                   'for annotation data, split into chunk files by positive or negative label as the file name'

import os
import csv
import pandas as pd

type = 'val'

def convert(domain, raw_data_path='./', save_path='../../concept_data/'):
    topic_f = os.path.join(raw_data_path, '{}.topics.tsv'.format(domain))
    topic_save_f = os.path.join(save_path, '{}_topics.tsv'.format(domain))

    """ convert topic """
    # convert topic_file to NLP data format, i.e. replace '\t' to '|'
    topic_data = []
    with open(topic_f, 'r') as f:
        for line in f:
            topic_data.append('|'.join(line.strip().split('\t')))

    with open(topic_save_f, 'w') as f:
        for line in topic_data:
            f.write(line + '\n')

    """ convert annotation """
    annotation_save_path = os.path.join(save_path, '{}_split'.format(domain))
    if not os.path.exists(annotation_save_path):
        os.mkdir(annotation_save_path)

    # CV and BIO annotation index are from 1, therefore, we will -1 the index
    for index in [0, 1, 2, 3, 4]:
        annotation_f = os.path.join(raw_data_path, domain, '{}.{}.csv'.format(type, index))
        pos, neg = [], []
        with open(annotation_f, 'r') as f:
            reader = csv.reader(f, delimiter=",")
            for i, line in enumerate(reader):
                line = [int(v) for v in line]
                # read & -1 the index
                line[0] -= 1
                line[1] -= 1

                # revert back to str
                if line[2] == 1:
                    line = ','.join([str(v) for v in line[:-1]])
                    pos.append(line)
                else:
                    line = ','.join([str(v) for v in line[:-1]])
                    neg.append(line)

        pos_save_f = os.path.join(annotation_save_path, '{}_pos_{}.txt'.format(type, index))
        neg_save_f = os.path.join(annotation_save_path, '{}_neg_{}.txt'.format(type, index))
        with open(pos_save_f, 'w') as f:
            for line in pos:
                f.write(line + '\n')
        with open(neg_save_f, 'w') as f:
            for line in neg:
                f.write(line + '\n')

    return


if __name__ == "__main__":
    # process CD data
    domain = 'CV'
    convert(domain=domain, raw_data_path='./', save_path='../../concept_data/')

    # process BIO data
    domain = 'BIO'
    convert(domain=domain, raw_data_path='./', save_path='../../concept_data/')
