# python

- supervised_openai_emb
    gain concept embedding from openai

- supervised_gcn
    concept_emb as init node embedding, dataset is built from split.

- tutorQA_build_graph
    a new dataset is built by ratio 1:1 from pos & neg.
    *result* new_graph_pair.csv

- cot_link_prediction
- zero_link_prediction

# res
- concept_emb llama/openai
- new_graph_pair
- cot/zs_*_msg.csv, the link prediction result message.
- cot/zs_*_pos_pair.csv, the pos label from link prediction result (y_{hat} == true)