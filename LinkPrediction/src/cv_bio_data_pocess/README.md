# Purpose

scripts in this folder are used to transform CV and BIO data into the format as the NLP data


* CV/{train/val/test}.{0/1/2/3/4}.csv
* CV.topics.tsv
* BIO/{train/val/test}.{0/1/2/3/4}.csv
* BIO.topics.tsv

After execute the scripts in this folder. The processed data will be stored in:

* CV
  * topics: /Text2Gen/concept_data/CV.topics.tsv
  * annotations: /Text2Gen/concept_data/CV_split/{positive, negative}_{0, 1, 2, ...}.txt
* BIO
  * topics: /Text2Gen/concept_data/BIO.topics.tsv
  * annotations: /Text2Gen/concept_data/BIO_split/{positive, negative}_{0, 1, 2, ...}.txt

__IMPORTANT: In order to align with the NLP data, ALL annotation index will -1, because index in NLP data is from 0, while they are from 1 in CV and BIO__



