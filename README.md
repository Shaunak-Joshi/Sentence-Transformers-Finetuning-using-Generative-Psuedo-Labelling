# **🤖FINETUNING SENTENCE TRANSFORMER BI-ENCODERS TO NFCORPUS🤖**

This repository provides you with a Python script that automates the process of training a specific Sentence Transformer model to defined knowledge base. The dataset selected is the NFCorpus dataset. 

## **WHY?**
To automate the process of finetuning a Sentence Transformer Embedding model to a dataset. Aims to produce (query, sentence, Cross-Encoder-Score) as a training sample. I could not find a golden truths file of sorts for this particular dataset; which can be used to train the model.  

This python script can be reproduced and modified for any BeIR dataset required. The pre-processing folder aims to replicate the MS-Marco golden truth values with the use of Semantic Search and comparing the retrieval using a Cross-Encoder. This provides you with a automated to script to convert query-to-corpus ground truths as provided in the NFCorpus dataset to query-to-sentence format, which is appended with a similarity score from 0 to 1. This makes it easy to make use of Cosine Similarity Loss and Embedding Similarity Evaluator for training and evaluation.

## **USAGE**
Make sure you have the NF Corpus installed. Link: https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/
1. Change Directory to the Pre-Processing Directory
   
   ```
   cd Pre-Processing
   ```
3. Run preprocessing.py
   
   ```
   python preprocessing.py
   ```
5. Now you will have three pickle files in the Pre-Processing Folder
6. Run finetune.py
   
   ```
   python finetune.py
   ```
