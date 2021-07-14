import re
import nltk
import heapq
import numpy as np
import pandas as pd
stopwords = nltk.corpus.stopwords.words('english')

def clean_and_tokenizer(text):
    text = re.sub(r'"', '', text)    
    sentence_list = nltk.sent_tokenize(text)
    return text, sentence_list
    
def frequency_finder(text):
    word_frequencies = {}
    for word in nltk.word_tokenize(text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/max_frequency)
        
    return word_frequencies
        

def sentence_scorer(sentence_list, word_frequencies):
    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 35:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    return sentence_scores                   

def summary_generator(sentence_scores):
    summary_sentences = heapq.nlargest(3, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)
    return summary
    #print(summary)
    
def summarizer(column_name):
    for index, row in data.iterrows():
        cleaned_and_tokenized_text, sentence_list = clean_and_tokenizer(row[column_name])
        word_frequencies = frequency_finder(cleaned_and_tokenized_text)
        sentence_scores = sentence_scorer(sentence_list, word_frequencies)
        summary = summary_generator(sentence_scores)
        row['predicted_summary'] = summary
        
    
if __name__ == '__main__':
    data = pd.read_csv('../dataset/test.csv')
    data['predicted_summary'] = ''
    data.fillna({'text': 'placeholder'}, inplace=True)
    summarizer('text')
    data.drop('summary', axis=1, inplace=True)
    data.to_csv('submission_nltk.csv', index=False)
    '''
    cleaned_and_tokenized_text, sentence_list = clean_and_tokenizer(text)
    print(cleaned_and_tokenized_text)
    print('\n')
    print(sentence_list)
    print('\n')
    word_frequencies = frequency_finder(cleaned_and_tokenized_text)
    print(word_frequencies)
    print('\n')
    sentence_scores = sentence_scorer(sentence_list, word_frequencies)
    print(sentence_scores)
    print('\n')
    summary = summary_generator(sentence_scores)
    print(summary)
    print('\n')
    '''