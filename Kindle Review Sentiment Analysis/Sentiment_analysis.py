import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import gensim
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# Load and preprocess the data
reviews = pd.read_csv('all_kindle_review.csv')
reviews = reviews[['reviewText', 'rating']]
reviews['rating'] = reviews['rating'].apply(lambda x: 0 if x < 3 else 1)
reviews['reviewText'] = reviews['reviewText'].str.lower()

# Remove non-alphanumeric characters and HTML tags, stopwords, and URLs
reviews['reviewText'] = reviews['reviewText'].apply(lambda x: re.sub('[^a-z A-Z 0-9]+', ' ', x))
reviews['reviewText'] = reviews['reviewText'].apply(lambda x: " ".join([word for word in x.split() if word not in stopwords.words('english')]))
reviews['reviewText'] = reviews['reviewText'].apply(lambda x: re.sub(r'(http|https|ftp|ssh)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', '', str(x)))
reviews['reviewText'] = reviews['reviewText'].apply(lambda x: BeautifulSoup(x, 'html.parser').get_text())
reviews['reviewText'] = reviews['reviewText'].apply(lambda x: " ".join(x.split()))

# Lemmatization
lemmatizer = WordNetLemmatizer()
corpus = reviews['reviewText']
words = [review.split() for review in corpus]
lemmatized_word = [[lemmatizer.lemmatize(word, pos="v") for word in review] for review in words]

# Train Word2Vec model
w2v_model = gensim.models.Word2Vec(lemmatized_word, vector_size=100, epochs=1000)

# Create feature vectors using Word2Vec averages
def avg_w2v(doc):
    vectors = [w2v_model.wv[word] for word in doc if word in w2v_model.wv.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(w2v_model.vector_size)

X_w2v = [avg_w2v(review) for review in lemmatized_word]
X_new_w2v = np.array(X_w2v)
y = reviews['rating'].to_numpy()

# Create DataFrame for features and target
df = pd.DataFrame(X_new_w2v)
df['Output'] = y

# Split features and target for training and testing
X = df.drop(columns=['Output'])
y = df['Output']
X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v = train_test_split(X, y, test_size=0.20, random_state=42)

# Train Naive Bayes model
nb_model_w2v = GaussianNB().fit(X_train_w2v, y_train_w2v)

# Predict and evaluate
y_pred_w2v = nb_model_w2v.predict(X_test_w2v)
print("W2V accuracy: ", accuracy_score(y_test_w2v, y_pred_w2v))
