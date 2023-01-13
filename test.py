from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

import pandas as pd

data_set = pd.read_csv(
    "C:\\Users\\ioana\\Documents\\GitHub\\Inteligenta-Artificiala-facultate\\Proiect\\mbti_1.csv")

data_set = data_set[:50]

types = list(data_set['type'])
posts = list(data_set['posts'])

# ######################
# Convert the sentences into a numerical representation
sentences = posts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
y = types

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest classifier on the training set
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

sentence_to_predict = vectorizer.transform(['This post explains me so well. I have been struggling so much, especially lately, with just feeling so lonely'])

print(classifier.predict(sentence_to_predict))

# Evaluate the model's performance
accuracy = classifier.score(X_test, y_test)
print('Accuracy:', accuracy)