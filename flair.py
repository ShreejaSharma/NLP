import pandas as pd
from flair.models import TextClassifier
from flair.data import Sentence
sia = TextClassifier.load('en-sentiment')

data_file = pd.read_csv('E:/MS Thesis/Dataset MS Thesis/ALL DATA SQL/All Data For all Wave/first_wave.csv')

def sentiment_Flair(x):
  sentence = Sentence(x)
  sia.predict(sentence)
  score = sentence.labels[0]
  if "POSITIVE" in str(score):
    return "POSITIVE"
  elif "NEGATIVE" in str(score):
    return "NEGATIVE"
  else:
    return "neutral"

data_file['sentiment_flair'] = data_file['text'].apply(lambda x: sentiment_Flair(x))
csv_data = data_file.to_csv('csvfile')
