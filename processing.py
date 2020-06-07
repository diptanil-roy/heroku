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
	author_list = ['Jane Austen', 'Lewis Carroll', 'Daniel Defoe', 'Charles Dickens', 'Sir Arthur Conan Doyle', 'Jack London', 'Joanne Rowling', 'Mary Shelley',
 'Louis Stevenson', 'Mark Twain', 'H.G. Wells', 'Oscar Wilde']
	y = int(model.predict(snip))
	z = round(model.predict_proba(snip)[0, y]*100, 1)

	return author_list[y], z

stream.title("Author Identification")
stream.header("By team data-dart")

'''
This is the winning submission to the [THE ERDŐS INSTITUTE May 2020 Data Science Boot Camp](https://www.erdosinstitute.org/code).

For more info on this project, please visit our [github page](https://github.com/data-dart/bookend) or watch this [youtube video](https://youtu.be/P1Sq7T9PvP0).
'''

stream.write("__Currently, we can only recognise the following authors:__ Jane Austen, Daniel Defoe, Charles Dickens, Sir Arthur Conan Doyle, Jack London, Joanne Rowling, Mary Shelley, Louis Stevenson, Mark Twain, H.G. Wells, and Oscar Wilde. ")

stream.write("*To identify the author with a high probability, please enter at least one chapter from the text.*")



snippet = stream.text_area("Enter text.", height=400)

submit = stream.button("Guess Author")

if submit:
	if (snippet!= ""):
		stream.write(guess_author(snippet)[0], "*with probability* ", guess_author(snippet)[1], "%")
		# stream.write()

	else:
		stream.write("__*You must enter some text.*__")