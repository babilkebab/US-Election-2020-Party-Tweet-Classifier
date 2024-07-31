import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
from sentence_transformers import SentenceTransformer
import numpy as np
import matplotlib.pyplot as plt

class TweetClassifier:
    def __init__(self, dataset):
        dataset = shuffle(dataset)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.train, self.test = train_test_split(dataset, test_size=0.2, stratify=dataset["label"], random_state=42)
        self.model = self._train_model()
        self._test_model()

    def _train_model(self):
        model = SVC(C=10, gamma=1, kernel="rbf")
        list_texts = self.train["tweet"].tolist()
        embedded_train = self.embedder.encode(list_texts)
        labels = np.array(self.train["label"])
        model.fit(embedded_train, labels)
        return model

    def _test_model(self):
        list_texts = self.test["tweet"].tolist()
        embedded_test = self.embedder.encode(list_texts)
        predictions = self.model.predict(embedded_test)
        cm = confusion_matrix(self.test["label"], predictions)
        ConfusionMatrixDisplay(cm).plot()
        plt.show()
        print(classification_report(self.test["label"], predictions))


    def save_model(self):
        with open('tweet_classifier.pkl','wb') as f:
            pickle.dump(self.model,f)

