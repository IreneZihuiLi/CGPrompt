data_path = ''
file_path = data_path + 'v5.freetext.txt'
large_file_path = data_path + 'v5.cleantext25.txt'


import re,os

def clean_line(line):
    # Remove HTML tags and strange characters
    cleaned_line = re.sub(r'<[^>]+>', '', line)  # Remove HTML tags
    cleaned_line = re.sub(r'[^\w\s]', '', cleaned_line)  # Remove non-alphanumeric characters

    # Remove short sentences or phrases
    if len(cleaned_line.split()) <= 25:  # Adjust the number as per your definition of 'short'
        return ''  # Return empty string for short lines
    return cleaned_line.strip()

with open(file_path, 'r') as file, open(large_file_path, 'w') as output_file:
    for line in file:
        cleaned_line = clean_line(line)
        if cleaned_line:  # Write only non-empty lines
            output_file.write(cleaned_line + '\n')

print("File cleaned and saved as:", large_file_path)


def split_file(file_path, lines_per_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(file_path, 'r') as file:
        count = 0
        file_count = 1
        current_content = []

        for line in file:
            count += 1
            current_content.append(line)

            if count == lines_per_file:
                write_chunk(current_content, file_count, output_folder)
                current_content = []
                count = 0
                file_count += 1

        # Write any remaining content
        if current_content:
            write_chunk(current_content, file_count, output_folder)

def write_chunk(content, file_count, output_folder):
    output_file_path = os.path.join(output_folder, f'chunk_{file_count}.txt')
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(content)


# Folder where the smaller files will be saved
output_folder = data_path+'v5_output_chunks25'  # Replace with your desired folder name

# Number of lines per small file
lines_per_file = 200

split_file(large_file_path, lines_per_file, output_folder)
print(f"File has been split and saved in folder: {output_folder}")


