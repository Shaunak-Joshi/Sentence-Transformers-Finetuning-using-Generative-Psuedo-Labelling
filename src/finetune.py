from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample, CrossEncoder
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datetime import datetime
import os
import pickle
import faiss
from tqdm import tqdm

class Train:
  def __init__(
    self,
    output_path: str= None,
    modelName: str = "all-mpnet-base-v2",
    device: str = "cuda",
    batch_size: int = 16,
    num_epochs: int = 20,
    save_best_model: bool= True,
    show_progres_bar: bool = True,
    ) -> None:
      if not (os.path.exists(os.path.join(os.getcwd(),'trainSamples.pkl'))):
        print("The current directory does not contain pickle files needed. Please move to that local directory")
      else:
        with open(os.path.join(os.getcwd(),'trainSamples.pkl'),"rb") as file:
            self.train = pickle.load(file)
        with open(os.path.join(os.getcwd(),'testSamples.pkl'),"rb") as file:
            self.test = pickle.load(file)
        with open(os.path.join(os.getcwd(),'devSamples.pkl'),"rb") as file:
            self.dev = pickle.load(file)

        self.model = SentenceTransformer(f"sentence-transformers/{modelName}",device=device)
        self.train_loss = losses.CosineSimilarityLoss(model=self.model)
        self.modelName = modelName
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_best_model = save_best_model
        self.show_progres_bar = show_progres_bar

        if output_path:
            self.output_path = output_path
        else:
            self.output_path = os.path.join(os.getcwd(),f"{modelName}_finetune_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}")

  def preProcessing(self):
    trainSamples = []
    testSamples = []
    devSamples = []

    for inputDict, samples in zip([self.train,self.test,self.dev],[trainSamples,testSamples,devSamples]):
      for simIndex in tqdm(inputDict):
        query = inputDict[simIndex][1]
        answer = inputDict[simIndex][0]
        samples.append(InputExample(texts=[query,answer],label=float(simIndex)))

    print(len(trainSamples),len(testSamples),len(devSamples))
    len(trainSamples)+len(testSamples)+len(devSamples)

    self.train_dataloader = DataLoader(trainSamples, shuffle=True, batch_size=self.batch_size)
    self.train_loss = losses.CosineSimilarityLoss(model=self.model)
    self.test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        testSamples, batch_size=self.batch_size, name=f"{self.modelName}_test_evaluation_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    )
    self.dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        devSamples, batch_size=self.batch_size, name=f"{self.modelName}_dev_evaluation_{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}"
    )
        
  def trainModel(
      self
      ) -> SentenceTransformer:
        
        self.preProcessing()  
        before = self.test_evaluator(self.model)
        self.model.fit(
            train_objectives = [(self.train_dataloader, self.train_loss)],
            evaluator= self.dev_evaluator,
            epochs = self.num_epochs,
            evaluation_steps = math.ceil(int(len(self.train_dataloader)/self.batch_size)),
            save_best_model = self.save_best_model,
            show_progress_bar=self.show_progres_bar,
            output_path = self.output_path
        )
        after = self.test_evaluator(self.model)
        return self.model,before,after

if __name__=="__main__":
    t = Train(device="cuda")
    model,evalBefore,evalAfter = t.trainModel()
    print(f"Evaluation results: {evalBefore} and {evalAfter}")