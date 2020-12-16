import pandas as pd
import numpy as np
#from transformers import pipeline
import streamlit as st
from numpy import argmax


# Download models ##########################

#Old from the website
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
#model = AutoModel.from_pretrained("facebook/bart-large-mnli")

# New -> from joe's sugestion:
# LARGE MODEL!!!!!!!!
#from transformers import AutoTokenizer, AutoModelForSequenceClassification
#tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
#model = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")

# pose sequence as a NLI premise and label as a hypothesis
# SMALLER MODEL!!!!!!!!
from transformers import AutoModelForSequenceClassification, AutoTokenizer
nli_model = AutoModelForSequenceClassification.from_pretrained('valhalla/distilbart-mnli-12-3')
tokenizer = AutoTokenizer.from_pretrained('valhalla/distilbart-mnli-12-3')
model='valhalla/distilbart-mnli-12-3'


st.title("One Shot Classifier")

# CPU based operations
#classifier = pipeline("zero-shot-classification", model='valhalla/distilbart-mnli-12-3')
# CPU based operations - lighter model "distilbart-mnli"
#classifier = pipeline("zero-shot-classification", model='valhalla/distilbart-mnli-12-3')
classifier = pipeline("zero-shot-classification")


#classifier = pipeline("zero-shot-classification")
# GPU based operations
#classifier = pipeline("zero-shot-classification", device=0) # to utilize GPU

#c32, c33 = st.beta_columns(2)
#
#
#with c33:

with st.beta_expander("candidate_labels 02", expanded=False):
    st.write("shoes")
    st.write("beaches")
    st.write("cars")

###################


MAX_LINES = 10

text = st.text_area("keyword, one per line.", height=150)
lines = text.split("\n")  # A list of lines

if len(lines) > MAX_LINES:
    st.warning(f"Maximum number of lines reached. Only the first {MAX_LINES} will be processed.")
    lines = lines[:MAX_LINES]

for line in lines:
    data = pd.DataFrame({'url':lines})

if not text:
    st.warning('Paste some keywords.')
    st.stop()


datalist = data.url.tolist()

datalistDeduped = [] 
Variable = [datalistDeduped.append(x) for x in datalist if x not in datalistDeduped] 

blocklist = {'',' ','-'}

candidate_labels = [x for x in datalistDeduped if x not in blocklist]

st.write("candidate_labels")
st.write(candidate_labels)

#st.stop()

import pandas as pd

#import csv
#import re
#import time
#import traceback
#from urllib.parse import urlparse
#import requests
#from bs4 import BeautifulSoup
#from langdetect import detect
#import matplotlib.pyplot as plt
#import seaborn as sns
#from numpy import argmax

from transformers import pipeline


#df = pd.read_csv('/content/google_results.csv', usecols=['content'], nrows=50)
#df = pd.read_csv('google_results.csv', usecols=['content'])

#df = pd.read_csv('macysTest50.csv', usecols=['content'])

df = pd.read_csv('macysTest50.csv', usecols=['content'], encoding = 'utf8')
#df = pd.read_csv('macysTest30.csv', usecols=['content'], encoding = 'utf8')

st.write(df)

#candidate_labels = ["renewable", "politics", "emission", "temperature", "emergency", "advertisment"]
#candidate_labels = ["SEO", "social", "link"]
#candidate_labels = ["spam", "advertisment", "informational"]
candidate_results = [0, 0, 0, 0, 0, 0]


for sent in df['content'].values:
    # To do multi-class classification, simply pass multi_class=True.
    # In this case, the scores will be independent, but each will fall between 0 and 1.
    res = classifier(sent, candidate_labels)


    SCORES = res["scores"]
    CLASSES = res["labels"]
    BEST_INDEX = argmax(SCORES)
    predicted_class = CLASSES[BEST_INDEX]
    predicted_score = SCORES[BEST_INDEX]

    if predicted_class == 'renewable' and predicted_score > 0.5:
        candidate_results[0] = candidate_results[0] + 1
    if predicted_class == 'politics' and predicted_score > 0.5:
        candidate_results[1] = candidate_results[1] + 1
    if predicted_class == 'emission' and predicted_score > 0.5:
        candidate_results[2] = candidate_results[2] + 1
    if predicted_class == 'temperature' and predicted_score > 0.5:
        candidate_results[3] = candidate_results[3] + 1
    if predicted_class == 'emergency' and predicted_score > 0.5:
        candidate_results[4] = candidate_results[4] + 1
    if predicted_class == 'advertisment' and predicted_score > 0.5:
        candidate_results[5] = candidate_results[5] + 1

    if res['scores'][0] > 0.5:
        st.write(sent)
        st.write(res['labels'])
        st.write(res['scores'])
        st.write()


#print(candidate_results)
st.write(candidate_results)
candidate_results

st.stop()


sequence = "boeing"
candidate_labels = ["boat", "car", "truck", "plane"]

res = classifier(sequence, candidate_labels)

res

st.stop()


##########################

c30, c31 = st.beta_columns(2)

with c30:
	with st.beta_expander("Bucket 01", expanded=False):
		st.write("boots")
		st.write("car")
		st.write("boat")

with c31:
	with st.beta_expander("Bucket 02", expanded=False):
		st.write("heels")
		st.write("SUV")
		st.write("ferry")


###################

#c32, c33 = st.beta_columns(2)
#
#with c32:
#	with st.beta_expander("candidate_labels 01", expanded=False):
#		st.write("shoe")
#		st.write("sea")
#		st.write("automotive")
#
#with c33:
#	with st.beta_expander("candidate_labels 02", expanded=False):
#		st.write("shoes")
#		st.write("beaches")
#		st.write("cars")

###################


MAX_LINES = 10

text = st.text_area("Label, one per line.", height=150)
lines = text.split("\n")  # A list of lines

if len(lines) > MAX_LINES:
    st.warning(f"Maximum number of lines reached. Only the first {MAX_LINES} will be processed.")
    lines = lines[:MAX_LINES]

for line in lines:
    data = pd.DataFrame({'url':lines})

if not text:
    st.warning('Paste some keywords.')
    st.stop()

datalist = data.url.tolist()

datalistDeduped = [] 
Variable = [datalistDeduped.append(x) for x in datalist if x not in datalistDeduped] 

blocklist = {'',' ','-'}

candidate_labels = [x for x in datalistDeduped if x not in blocklist]

st.write("candidate_labels")
st.write(candidate_labels)
#candidate_labels = ["shoe", "sea", "automotive"]




df2 = pd.DataFrame(np.array([["boots", 2,3 ], ["boat", 5, 6], ["car", 8, 9]]),columns=['keyword', 'b', 'c'])

st.write(df2)

sequence = df2['keyword']

#candidate_labels = ["shoe", "sea"]

results = classifier(sequence, candidate_labels)

#st.write(results)

dfnew = pd.DataFrame(results)
st.write(dfnew)

#st.stop()


