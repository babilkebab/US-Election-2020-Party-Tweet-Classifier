import pandas as pd
import lexicon_update
from SentimentAnalyzer import SentimentAnalyzer


biden_ds = pd.read_csv("ds/hashtag_joebiden.csv", lineterminator='\n', parse_dates=True)
trump_ds = pd.read_csv("ds/hashtag_donaldtrump.csv", lineterminator='\n', parse_dates=True)
init_complete_ds = pd.concat([biden_ds, trump_ds])



labeled_biden_ds = SentimentAnalyzer(biden_ds, "biden").analyze()
labeled_trump_ds = SentimentAnalyzer(trump_ds, "trump").analyze()


complete_ds = pd.concat([labeled_biden_ds, labeled_trump_ds])
stop_words = lexicon_update.stopwords_no_neg()
complete_ds.to_csv("ds/complete_ds.csv", index=False)
