import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


Data = pd.read_csv("news.csv")
labels = Data.label
x_train,x_test,y_train,y_test=train_test_split(Data['text'], labels, test_size=0.2, random_state=7)


