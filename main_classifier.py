import pandas as pd
from src.TweetClassifier import TweetClassifier


if __name__ == "__main__":
    final_ds = pd.read_csv("ds/complete_ds.csv", parse_dates=True)
    final_ds["label"] = final_ds["label"].replace({"GOP": 0, "DEM": 1})
    
    classifier = TweetClassifier(final_ds)
    classifier.save_model()
