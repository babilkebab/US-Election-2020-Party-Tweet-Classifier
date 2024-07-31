from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import lexicon_update
import re
import lib_util
from sentence_transformers import SentenceTransformer



class SentimentAnalyzer:
    def __init__(self, dataset, candidate):
        self.candidate = candidate
        self.analyzer = SentimentIntensityAnalyzer()
        self._lexicon()
        self.dataset = dataset
        self.dataset = self._preprocess()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")


    def analyze(self):
        labels = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
        labeled_dataset = self.dataset.copy()
        labeled_dataset['label'] = self.dataset['tweet'].apply(lambda tweet: self.analyzer.polarity_scores(tweet)['compound'])

        i = 0
        for row in labeled_dataset.iterrows():
            row[1]["label"] = lib_util.compound(row[1]["tweet"], row[1]["label"], self.candidate)
            i+=1

        labeled_dataset['label'] = pd.cut(labeled_dataset['label'], bins=5, labels=labels)
        labeled_dataset = labeled_dataset.where(labeled_dataset['label'] != "Neutral").dropna()
        labeled_dataset = labeled_dataset.where(labeled_dataset['label'] != "Positive").dropna()
        labeled_dataset = labeled_dataset.where(labeled_dataset['label'] != "Negative").dropna()
        if self.candidate == "biden":
            labeled_dataset["label"] = labeled_dataset["label"].replace({"Very positive": "DEM"})
            labeled_dataset["label"] = labeled_dataset["label"].replace({"Very negative": "GOP"})
        else:
            labeled_dataset["label"] = labeled_dataset["label"].replace({"Very positive": "GOP"})
            labeled_dataset["label"] = labeled_dataset["label"].replace({"Very negative": "DEM"})
        return labeled_dataset


    def _lexicon(self):
        if self.candidate == "biden":
            new_words = lexicon_update.new_words_biden()
        else:
            new_words = lexicon_update.new_words_trump()
        self.analyzer.lexicon.update(new_words)

    def _preprocess(self):
        stop_words = lexicon_update.stopwords_no_neg()
        self.dataset = self.dataset.drop_duplicates()
        self.dataset["single_author"] = self.dataset["source"].apply(lambda source: lib_util.filter_papers(source))
        self.dataset = self.dataset.drop(self.dataset[self.dataset['single_author'] == False].index)
        country_condition = (self.dataset['country'] != "United States of America") & (self.dataset['country'] != "United States") & (self.dataset['country'] != None)
        self.dataset = self.dataset.drop(self.dataset[country_condition].index)
        self.dataset["lang"] = self.dataset["tweet"].apply(lambda tweet: lib_util.detect_lang(tweet))
        self.dataset = self.dataset.drop(self.dataset[self.dataset['lang'] != "en"].index)
        self.dataset['tweet'] = self.dataset['tweet'].apply(lambda tweet: tweet.lower())
        self.dataset = self.dataset.replace(r"http\S+", "",regex=True).astype(str)

        self.dataset["tweet"] = self.dataset["tweet"].apply(lambda tweet: re.sub(r'[^a-zA-Z\s]', '', tweet))
        self.dataset["tweet"] = self.dataset["tweet"].apply(lambda tweet: lib_util.remove_entities(tweet))
        self.dataset["tweet"] = self.dataset["tweet"].apply(lambda tweet: lib_util.lemmatizer(tweet))
        self.dataset['tweet'] = self.dataset['tweet'].apply(lambda tweet: " ".join([word for word in tweet.split() if word not in stop_words]))

        self.dataset["user_description"] = self.dataset["user_description"].apply(lambda x: re.sub(r'\s+', ' ', str(x)))
        self.dataset["user_description"] = self.dataset["user_description"].apply(lambda x: re.sub(r'\n+', ' ', str(x)))
        self.dataset["user_description"] = self.dataset["user_description"].apply(lambda x: lib_util.remove_entities(str(x)))
        self.dataset["user_description"] = self.dataset["user_description"].apply(lambda x: x.lower())
        self.dataset["user_description"] = self.dataset["user_description"].apply(lambda user_description: " ".join([word for word in user_description.split() if word not in stop_words]))
        self.dataset["user_description"] = self.dataset["user_description"].apply(lambda user_description: lib_util.lemmatizer(user_description))
        self.dataset = self.dataset.drop(columns=["single_author"])

        
        if self.candidate == "biden":
            self.dataset = self.dataset.drop(self.dataset[self.dataset['tweet'].str.contains("trump")].index)
        else:
            self.dataset = self.dataset.drop(self.dataset[self.dataset['tweet'].str.contains("biden")].index)
        return self.dataset
