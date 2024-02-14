# Process
- 2_new_graph.py
    sampling 1:1 pos:neg from final_new_annotation.csv
    got generated_pair.csv
- 3_link_prediction.py
    request openai from link prediction based on generated_pair.csv
    store msg on link_msg_result
    store all y_pred == pos into pos_neg_recover.csv
- 4

# Result_pair
- cot-1-1-result.csv: GPT-4-CoT message storage.
- generated_pair.csv: 1:1 pos & neg from final_new_annotation.csv
- link_msg.csv: GPT-4 message storage
- pos_neg_recover.csv: 1:1 GPT 4 best result. ☑️
- result_pair.csv: GPT-4-CoT predicition's pos pair only
