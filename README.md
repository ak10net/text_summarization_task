# Text summarization case study
*Extractive and Abstractive text summarization*

Definitions:-
-  *Extractive*
Extractive Summarization essentially involves extracting particular pieces of text (usually sentences) based on predefined weights assigned to the important words where the selection of the text depends on the weights of the words in it. Usually, the default weights are assigned according to the frequency of occurrence of a word. Summary contains sentences from original text.

-  *Abstractive*
Abstractive Summarization includes heuristic approaches to train the system in making an attempt to understand the whole context and generate a summary based on that understanding. Summary may or may not contain sentences from original text.


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
-  'abstractive_summarizer.py' tummarization using deep learning and transformers (Not working)
-  Based on tensorflow tutorial on text generation using transformer

-  The core idea behind the Transformer model is self-attention—the ability to attend to different positions of the input sequence to compute a representation   of that sequence. Transformer creates stacks of self-attention layers and is explained below in the sections Scaled dot product attention and Multi-head    attention.

-  The transformer is an auto-regressive model: it makes predictions one part at a time, and uses its output so far to decide what to do next.
   During training this example uses teacher-forcing (like in the text generation tutorial). Teacher forcing is passing the true output to the next time step regardless of what the model predicts at the current time step.As the transformer predicts each word, self-attention allows it to look at the previous words in the input sequence to better predict the next word.
   
-  
 	**Logic**
	-  Cleaning, tokenization by fitting keras preprocessing text tokenizer on text and summaries
	-  Padding and truncation of text and summaries for uniformity
	-  Creating positional encoding of text to give model some information about the relative position of words in the sentence
	-  Padding mask to ignore padding added to sentences so that model does not treat padding as input and look ahead mask to ignore words coming after the    current word to be used for prediction
	-  Attention funciton used by transformer uses Query, key and value
	-  Multi-head attention to split inputs into multiple heads and calculate attention weights using scaled dot product and concat all heads and final linear   layer.
	-  Each multi-head attention block gets three inputs as Q, K, V. These are put through linear dense layers and split up into multiple heads
	-  The input sentence is passed through N encoder layers that generates an output for each word/token in the sequence.
	-  The decoder attends on the encoder's output and its own input (self-attention) to predict the next word.
	-  Encoder and decoder block to obtain input embedding which are added to positional encodings
		-  Encoder consists of : Input embedding , positional encodings andN encoded layers
		-  Output of encoder is input for decoder
		-  Decoder consists  of: Output embeddings, poisitonal encoding and N decoder layers
	-  The output of decoder is input to the final layer
	-  Transformer model creation using atttention and encoder decoder blocks
		-  Stacking encoder decoder under transformer class that inherits from tf.keras.model
	-  Model training with set of hyperparameters
	-  Evaluation step
		-  Input sentence is tokenized and fed to encoder
		-  Decoder input is intialized to [Start] token
		-  Padding and looks ahead masks are calculated
		-  Decoder outputs predictions by looking at encoder output and its out output (self attention)
		-  Model makes next word prediction for each word in output.
	-  Inference step to provide previous prediction as input to decoder
	- Keep appending output to decoder input untils sequence reaches maxlen or predicted word is stop token
	


**Refrences**

*Theory on text summarization*

[Link](https://medium.com/luisfredgs/automatic-text-summarization-with-machine-learning-an-overview-68ded5717a25)

[Link](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/)

*Abstractive summarization using transformer*

[Link](https://www.tensorflow.org/text/tutorials/transformer)

[Link](https://medium.com/swlh/abstractive-text-summarization-using-transformers-3e774cc42453)

