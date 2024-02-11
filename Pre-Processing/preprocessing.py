from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample, CrossEncoder
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datetime import datetime
import os
import pickle
import faiss
import csv
import json
from tqdm import tqdm
from .vectorstore import Vectorstore
import pickle


class LoadDataset:
    def __init__(self):
        self.path = 'path-to-nfcorpus-folder'
        self.files = ['corpus.jsonl','queries.jsonl',['qrels/dev.tsv','qrels/test.tsv','qrels/train.tsv']]

    def load_corpus(self):
        corpus = {}
        num_lines = sum(1 for i in open(os.path.join(self.path,self.files[0]), 'rb'))
        with open(os.path.join(self.path,self.files[0]), encoding='utf8') as fIn:
            for line in tqdm(fIn, total=num_lines):
                line = json.loads(line)
                corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }
        return corpus

    def load_queries(self):
        queries = {}
        with open(os.path.join(self.path,self.files[1]), encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                queries[line.get("_id")] = line.get("text")
        return queries

    def load_qrels(self):
        dev = {}
        test = {}
        train = {}
        for i,qrels in zip(self.files[2],[dev,test,train]):
          reader = csv.reader(open(os.path.join(self.path,i), encoding="utf-8"),
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
          next(reader)
          l = 0
          for id, row in enumerate(reader):
              l += 1
              query_id, corpus_id, score = row[0], row[1], int(row[2])

              if query_id not in qrels:
                  qrels[query_id] = {corpus_id: score}
              else:
                  qrels[query_id][corpus_id] = score
        return train,test,dev

    def getDataset(self):
        return self.load_corpus(),self.load_queries(), self.load_qrels()
    


if __name__=="__main__":
    l = LoadDataset()
    
    corpus = l.load_corpus()
    queries = l.load_queries()
    train,test,dev = l.load_qrels()
    model = SentenceTransformer("all-miniLM-L6-v2",device="cuda")
    
    trainSamples = {}
    testSamples = {}
    devSamples = {}
    
    for qrels,samples in zip([train,test,dev],[trainSamples,testSamples,devSamples]):
        for qid in tqdm(qrels):
            answers = {}
            for pid in qrels[qid]:
                para = corpus[pid]['text'].split('.')
                v = Vectorstore(model,para)
                answerDict = v.vectorStore(model,para,queries[qid])
                for i in answerDict:
                    answers[i] = answerDict[i]
            values = sorted(list(answers.keys()),reverse=True)
            samples[values[0]] = (answers[values[0]],queries[qid])
    
    
    for name,dic in zip(['trainSamples','testSamples','devSamples'],[trainSamples,testSamples,devSamples]) :
        with open(os.path.join(os.getcwd(),name)+'.pkl',"wb") as file:
            pickle.dump(dic,file)
