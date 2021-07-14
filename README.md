# Text summarization case study
*Extractive and Abstractive text summarization*

This repo contains following folders:-

## Dataset
-  Unzipped Train and Test folder ( unzip the dataset here )
-  'csv_generator.py' that generates train and test csv from folders. It takes 'train' or 'test' as argument
-  To generate dataset as csv run 'python csv_generator.py train or test'
	
## Extractive
-  'extractive_nltk.py' generates summaries based on word frequency and sentence scores
-  
	**Logic**
	-  Pre-process the text: remove stop words
	-  Tokenize whole text and creates weighted word frequeny dictionary
	-  For each sentence it calculates the overall score by adding word weights
	-  Selected the top three sentences with highest sentence score
	-  To generate summary on test set run 'python extractive_nltk.py'
		
- 'extractive_gensim.py' generates summaries using 'summarize' module of 'gensim' library which is based on 'TextRank' an unsupervised algorithm based on weighted-graphs. 
- 
	**Logic**
	-  Pre-process the text: remove stop words and stem the remaining words.
	-  Create a graph where vertices are sentences.
	-  Connect every sentence to every other sentence by an edge. The weight of the edge is how similar the two sentences are.
	-  Run the PageRank algorithm on the graph.
	-  Pick the vertices(sentences) with the highest PageRank score
	-  To generate summary on test set run 'python extractive_gensim.py'
		
-  'submission_nltk.csv' and 'submission_gensim.csv' output files
	
## Abstractive
-  'abstractive_summarizer.py' tummarization using deep learning and transformers (does not work)
-  
 	**Logic**
	



**Theory on text summarization**

[theory](https://medium.com/luisfredgs/automatic-text-summarization-with-machine-learning-an-overview-68ded5717a25)

[theory](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/)

**References used to build abstractive summarization using transformer**

[Link](https://www.tensorflow.org/text/tutorials/transformer)

[Link](https://medium.com/swlh/abstractive-text-summarization-using-transformers-3e774cc42453)

[Link](https://huggingface.co/transformers/model_doc/pegasus.html)
