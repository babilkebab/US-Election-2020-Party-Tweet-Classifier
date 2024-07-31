from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from langdetect import detect
import langdetect
import string
from lexicon_update import gop_lexicon, dem_lexicon


def detect_lang(tweet):
    try:
        return detect(tweet)
    except langdetect.lang_detect_exception.LangDetectException as e:
        return None


def filter_papers(tweet_source):
    if tweet_source != "Twitter for iPhone" and tweet_source != "Twitter for Android" \
        and tweet_source != "Twitter Web App" and tweet_source != "Twitter for iPad" and tweet_source != "Twitter Web Client": 
        return False
    
def remove_entities (text):
    entity_prefixes = ['@', '#']
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator, '')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
                if word[0] not in entity_prefixes:
                    words.append(word)
    return ' '.join(words)

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def lemmatizer(string):
    wl = WordNetLemmatizer()
    word_pos_tags = nltk.pos_tag(word_tokenize(string))
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] 
    return " ".join(a)


def compound(text, row_compound, candidate):
    pts_dem = 0
    text = text.lower()
    for word in dem_lexicon():
        if word in text:
            pts_dem += dem_lexicon()[word]
    pts_gop = 0
    for word in gop_lexicon():
        if word in text:
            pts_gop += gop_lexicon()[word]
    if candidate == "biden":
        if pts_dem > pts_gop:
            if pts_dem > 0.5:
                row_compound += 0.5
            else:
                row_compound += pts_dem
        elif pts_dem < pts_gop:
            if pts_gop > 0.5:
                row_compound -= 0.5
            else:
                row_compound -= pts_gop
    else:
        if pts_gop > pts_dem:
            if pts_gop > 0.5:
                row_compound += 0.5
            else:
                row_compound += pts_gop
        elif pts_gop < pts_dem:
            if pts_dem > 0.5:
                row_compound -= 0.5
            else:
                row_compound -= pts_dem

    if row_compound > 1.0:
        row_compound = 1.0
    elif row_compound < -1.0:
        row_compound = -1.0
    
    return row_compound



