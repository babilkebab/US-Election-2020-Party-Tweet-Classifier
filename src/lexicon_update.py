from nltk.corpus import stopwords



def new_words_biden():
    return {
                    "jomentia": -4.0,
                    "hunter": -4.0,
                    "hunterbiden": -4.0,
                    "laptop": -4.0,
                    "corrupt": -4.0,
                    "dementia": -4.0,
                    "emails": -4.0,
                    "sleepy": -4.0,
                    "creepy": -4.0,
                    "corruption": -4.0,
                    "bidenemails": -4.0,
                    "maga": -4.0,
                    "retirement": -4.0,
                    "retire": -4.0,
                    "blm": 2.0,
                    "blacklivesmatter": 2.0,
                    "votebiden": 4.0,
                    "ukrainian": -4.0,
                    "ukraine": -4.0,
                    "bidencorrupt": -4.0,
                    "joebideniscorrupt": -4.0,
                    "votebiden": 4.0,
                    "bidenharristosaveamerica": 4.0,
                    "bidenharris2020": 4.0,
                    "presidentelectjoebiden": 4.0,
                    "bidencrimefamily": -4.0,
                    "votebluetosaveamerica": 4.0,
                    "biden2020": 4.0,
                }

def new_words_trump():
    return{
                "corrupt": -3.0,
                "fraud": -3.0,
                "rigged": -3.0,
                "fake": -3.0,
                "scam": -3.0,
                "cheat": -3.0,
                "steal": -3.0,
                "liar": -3.0,
                "thief": -3.0,
                "criminal": -3.0,
                "crook": -3.0,
                "con": -3.0,
                "conman": -3.0,
                "censorship": -3.0,
                "maga": 4.0,
                "impeach": -3.0,
                "votetrump": 4.0,
                "putin": 4.0,
                "trumpcrimefamily": -4.0,
                "trumpusaracist": -4.0,
                "gopbetrayedamerica": -4.0,
                "votebluetosaveamerica": -4.0,
                "trumpvirus": -4.0,
                "trumpliespeopledie": -4.0,
                "trumpisacriminal": -4.0,
                "trumpispathetic": -4.0,
                "pandemic": -4.0,
                "covid": -4.0,
                "coronavirus": -4.0,
                "virus": -4.0,
                "trumpvirus": -4.0,
                "trump2020": 4.0,
            }

def stopwords_no_neg():
    stop_words = stopwords.words('english')
    list_neg = ["but", "not", "no", "ain", "aren", "couldn", "didn", "doesn", "hadn", 
            "hasn", "haven", "isn", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", 
            "weren", "won", "wouldn", "don"]
    l = len(list_neg)
    for i in range(3, l):
        list_neg.append(list_neg[i] + "'t")
    stop_words = list(set(stop_words) - set(list_neg))
    return stop_words



def gop_lexicon():
    return {
        "army": 0.05,
        "navy": 0.05,
        "republican": 0.5,
        "trump": 0.1,
        "donald": 0.1,
        "trump2020": 0.2,
        "business": 0.03,
        "conservative": 0.1,
        "conservationist": 0.1,
        "catholic": 0.1,
        "christ": 0.1,
        "christian": 0.1,
        "entrepreneur": 0.05,
        "family": 0.075,
        "gop": 0.2,
        "jesus": 0.1,
        "god": 0.1,
        "guns": 0.05,
        "maga": 0.2,
        "neverbiden": 0.2,
        "pastor": 0.075,
        "patriot": 0.1,
        "patriots": 0.1,
        "retrumplican": 0.2,
        "red": 0.1,
        "tradition": 0.05, 
        "traditional": 0.05,
        "traditionalist": 0.1,
        "train": 0.05,
        "truth": 0.075,
        "veteran": 0.075,
        "votetrump": 0.5,
        "votered": 0.5
    }

def dem_lexicon():
    return {
        "activist": 0.1,
        "atheist": 0.075,
        "blm": 0.1,
        "biden": 0.1,
        "bidenharris": 0.1,
        "blacklivesmatter": 0.1,
        "blue": 0.1,
        "climate": 0.05,
        "democrat": 0.5,
        "feminist": 0.1,
        "feminism": 0.1,
        "gay": 0.1,
        "healthcare": 0.05,
        "labor": 0.05,
        "lgbtq": 0.1,
        "liberal": 0.075,
        "peace": 0.05,
        "phd": 0.025,
        "progressive": 0.1,
        "proud": 0.05,
        "racism": 0.075,
        "votebiden": 0.2,
        "voteblue": 0.5,
        "votebluenomatterwho": 0.5,
        "votetrumpout": 0.5,
        "nevertrump": 0.5,
        "rights": 0.075,
        "right": 0.075,
        "trans": 0.1,
    }
