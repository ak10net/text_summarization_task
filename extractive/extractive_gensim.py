import pandas as pd
from gensim.summarization import summarize

#data = pd.read_csv('dump.csv')

def summarizer(column_name):
    for index, row in data.iterrows():
        row['predicted_summary'] = summarize(row[column_name], word_count=100)

#input_text = data.iloc[0]['text']
#print('Input text: ')
#print(input_text)
#print('\n')
#print('\n')
#print(summarize(input_text, word_count=100))


if __name__ == '__main__':
    data = pd.read_csv('../dataset/test.csv')
    data['predicted_summary'] = ''
    data.fillna({'text': 'placeholder'}, inplace=True)
    summarizer('text')
    data.drop('summary', axis=1, inplace=True)
    data.to_csv('submission_gensim.csv', index=False)