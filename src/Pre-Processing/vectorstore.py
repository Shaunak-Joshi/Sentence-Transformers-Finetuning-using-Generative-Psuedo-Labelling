import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder

class Vectorstore:
  def __init__(self,model,documents):
    self.index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
    self.embeddings = model.encode(documents)
    faiss.normalize_L2(self.embeddings)
    self.index.add(self.embeddings)
    self.ce = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device="cuda")

  def sigmoid(self,x):
    return 1/(1+np.exp(-x))

  def vectorStore(self,model,documents,query):
      tempDict = {}
      returnDict = {}

      search_vector = model.encode(query)
      _vector = np.array([search_vector])
      faiss.normalize_L2(_vector)
      distances, ann = self.index.search(_vector, k=10)

      for i,j in zip(distances[0],ann[0]):
        tempDict[documents[j]] = i

      for i in tempDict:
        returnDict[self.sigmoid(self.ce.predict([query,i]))] = i

      values = sorted(list(returnDict.keys()),reverse=True)[:2]

      answer = {}
      for i in values:
        answer[i] = returnDict[i]
      return answer