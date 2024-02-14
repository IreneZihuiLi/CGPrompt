

# input_file = 'TutorQA_test/M1_4_summary.txt'


# def process_line(line):
#     """Extracts title, description, and concepts from a line."""
#     parts = line.split('Description:')
#     title = parts[0].split('Title:')[1].strip()
#     description, concepts = parts[1].split('Concepts:')
#     description = description.strip()
#     concepts = concepts.strip()
#     return title, description, concepts
#
# def convert_to_tsv(input_file, output_file):
#     with open(input_file, 'r') as file:
#         lines = file.readlines()
#
#     with open(output_file, 'w') as file:
#         # Write the header
#         file.write("Title\tDescription\tConcepts\n")
#
#         for line in lines:
#             title, description, concepts = process_line(line)
#             file.write(f"{title}\t{description}\t{concepts}\n")
#
# # Example usage
#
# output_file = 'TutorQA_test/M1_4_summary_clear.tsv'
# convert_to_tsv(input_file, output_file)


import csv

input_file = 'TutorQA_test/M1_4_new_batch.txt'

# Sample data (normally this would come from your txt file)
# data = """
# Title: "Voice-Activated Personal Health Assistant"
# Description: Develop a voice-activated application that understands and responds to health-related queries. This system could provide advice on common medical conditions, medication dosages, and first aid tips, based on user's spoken input.
# Relevant Concepts: speech recognition; natural language processing; intent classification; health informatics; dialogue systems.
# Title: "Real-Time Social Media Trend Analysis"
# Description: Create a tool that analyzes social media posts in real-time to identify emerging trends, popular topics, and public opinions. This project could include sentiment analysis and topic modeling to provide insights into current social dynamics.
# Relevant Concepts: real-time data processing; sentiment analysis; topic modeling; social network analysis; data visualization.
# """

with open(input_file, 'r', encoding='utf-8') as file:
    data = file.read()
# Splitting the data into records
records = data.strip().split("\n")

# Parsing each record
# Parsing each record
parsed_records = []
title = description = concepts = ""
for record in records:
    if record:
        # Extracting the title, description, and concepts
        lines = record.split("\n")
        for line in lines:
            print (line)
            if "Project Title:" in line:
                title = line.split(':')[1].strip()[1:-1].replace('\n','')
            elif "Description:" in line:
                description = line.split('Description: ')[1].strip().replace('\n','')
            elif "Concepts:" in line:
                concepts = line.split('Concepts: ')[1].strip().replace('\n','')
        # import pdb;pdb.set_trace()
        if len(title) > 1 and len(description) > 1 and len(concepts) > 1:
            parsed_records.append([title, description, concepts])
            title = description = concepts = ""

# Writing to a TSV file

tsv_filename = 'TutorQA_test/M1_4_new_batch_clean.tsv'
with open(tsv_filename, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter='\t')
    writer.writerow(['Title', 'Description', 'Relevant Concepts'])  # Writing the header
    writer.writerows(parsed_records)

