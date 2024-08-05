from transformers import AutoTokenizer, AutoModel
import torch

class InferlessPythonModel:
    
    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
        self.model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5').to("cuda")
        self.model.eval()

    def infer(self, inputs):
        sentences = inputs["sentences"]

        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to("cuda")
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]

        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)

        return {"embeddings":sentence_embeddings.tolist()}
    
    def finalize(self):
        self.model = None