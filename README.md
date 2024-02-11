### **FINETUNING SENTENCE TRANSFORMER BI-ENCODERS TO NFCORPUS**

This repository provides you with a Python script that automates the process of training a specific Sentence Transformer model to defined knowledge base. The dataset selected is the NFCorpus dataset. 

This python script can be reproduced and modified for any BeIR dataset required. The pre-processing folder aims to replicate the MS-Marco golden truth values with the use of Semantic Search and comparing the retrieval using a Cross-Encoder. This provides you with a automated to script to convert query-to-corpus ground truths as provided in the NFCorpus dataset to query-to-sentence format, which is appended with a similarity score from 0 to 1. 

This makes it easy to make use of Cosine Similarity Loss and Embedding Similarity Evaluator for training and evaluation.
