import sys
import os
import re
import csv
import pandas as pd


project_dir = 'C:/Users/ankit.dubey/Desktop/text_summarization'
train_dir = 'dataset/train'
test_dir = 'dataset/test'
train_dir_path  = os.path.join(project_dir, train_dir)
test_dir_path = os.path.join(project_dir, test_dir)
directory_path = ''
columns = ['file_id', 'text', 'summary']
data = pd.DataFrame(columns=columns)
file_names = []
texts = []
summaries = []

def dataloader(directory_path):
    for i, filename in enumerate(os.listdir(directory_path)):
        #print(filename)
        with open(os.path.join(directory_path,filename), 'r', newline='\n', encoding='utf-8') as f:
            file_names.append(filename)
            #data.loc[i, 'file_id'] = filename
            reader = csv.reader(f.read().splitlines(), delimiter='\t')
            lines = []
            for row in reader:
                lines.append(row)
            #lines = f.readlines()
            flat_lines = [item for sublist in lines for item in sublist]
            lines = ' '.join(x for x in flat_lines)
            #lines = re.sub("[^a-zA-Z]",' ',str(lines))
            text_pattern = re.compile(r"[^@]*")
            #data.loc[i, 'text'] = re.findall(text_pattern, lines)
            texts.append(re.findall(text_pattern, lines)[0])
            summary_pattern = re.compile(r"@highlight\n{0,}.*")
            summaries.append(str(re.sub('@highlight', ' ', str(re.findall(summary_pattern, lines)))))
    #print(texts)
    #print(summaries)

    data['file_id'] = file_names
    data['text'] = texts     
    data['summary'] = summaries
    data.to_csv(directory_path + '.csv', index=False, encoding='utf-8')         
    
    
if __name__ == "__main__":
    #folder_argument = sys.argv[1]
    if len(sys.argv) == 2 and sys.argv[1] == 'train':
        directory_path = train_dir_path
    elif len(sys.argv) == 2 and sys.argv[1] == 'test':
        directory_path = test_dir_path
    else:
        print('please provide the input folder argument for csv generation')
    dataloader(directory_path)