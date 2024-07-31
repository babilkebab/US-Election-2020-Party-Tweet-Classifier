import gensim.downloader as api
from nltk.corpus import stopwords
import re, lib_util
import numpy as np

class EmbeddingModel:
    def __init__(self, dataset, training_param, size=200):
        self.dim = size
        self.model = api.load('glove-twitter-200')
        self.training_param = training_param
        self.dataset = self._preprocess(dataset)



    def _preprocess(self, dataset):
        stop_words = stopwords.words('english')
        dataset = dataset.dropna()
        dataset = dataset.drop_duplicates()
        dataset[self.training_param] = dataset[self.training_param].apply(lambda text: text.lower())
        dataset[self.training_param] = dataset[self.training_param].apply(lambda text: re.sub(r'[^a-zA-Z\s]', '', text))
        dataset[self.training_param] = dataset[self.training_param].apply(lambda text: lib_util.remove_entities(text))
        dataset[self.training_param] = dataset[self.training_param].apply(lambda text: lib_util.lemmatizer(text))
        dataset[self.training_param] = dataset[self.training_param].apply(lambda text: " ".join([word for word in text.split() if word not in stop_words]))
        return dataset
    

    def _generate_embeddings(self, dataset):
        text_embeddings = []
        for text in dataset:
            sentence_embedding = []
            if type(text) is float:
                text = "*"
            for word in text:
                try:
                    sentence_embedding.append(self.model[word])
                except KeyError:
                    sentence_embedding.append([0]*self.dim)
            text_embeddings.append(np.mean(sentence_embedding, axis=0))
        return text_embeddings 

    def get_embeddings(self, texts):
        if type(texts) is float:
            texts = texts.astype(str)
        embeddings = self._generate_embeddings(texts)
        return embeddings

    



