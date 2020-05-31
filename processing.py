import pickle
import streamlit as stream
import pandas as pd
from src.model import LexicalFeatures, BOWFeatures, SyntacticFeatures, NGramFeatures
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer, LabelEncoder, MaxAbsScaler

@stream.cache

def guess_author(snip: str):

	model = pickle.load(open('model_fit','rb'))
	author_list = ['austen', 'carroll', 'defoe', 'dickens', 'doyle', 'london', 'rowling', 'shelley',
 'stevenson', 'twain', 'wells', 'wilde']
	y = int(model.predict(snip))

	return author_list[y]

stream.title("Enter your text snippet. Make sure it's long enough.")

snippet = stream.text_input('Enter text')

submit = stream.button("Guess Author.")

if submit:
	stream.write(guess_author(snippet))