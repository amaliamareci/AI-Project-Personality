import sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from preprocess import preprocess, preprocess_sentence
import pandas as pd
import re



class RandomForest:
    def __init__(self, X_train, X_test, Y_train, Y_test, vectorizer):
        self.vectorizer = vectorizer
        accuracies = {}
        # Random Forest
        self.random_forest = RandomForestClassifier(
            n_estimators=100, random_state=1)
        self.random_forest.fit(X_train, Y_train)

        # make predictions for test data
        Y_pred = self.random_forest.predict(X_test)
        print(Y_pred)

        # evaluate predictions
        self.accuracy = accuracy_score(Y_test, Y_pred)
        # accuracies['Random Forest'] = accuracy* 100.0
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    def get_accuracy(self):
        return round(self.accuracy * 100.0, 2)

    def predict_text(self, sentence):
        sentence = preprocess_sentence(sentence)
        train = self.vectorizer.transform([sentence])
        return self.random_forest.predict(train)


class SGD:
    def __init__(self, X_train, X_test, Y_train, Y_test, vectorizer):
        self.vectorizer = vectorizer
        accuracies = {}
        # Gradient Descent
        self.sgd = SGDClassifier(max_iter=5, tol=None)
        self.sgd.fit(X_train, Y_train)

        Y_pred = self.sgd.predict(X_test)
        print(Y_pred)

        # evaluate predictions
        self.accuracy = accuracy_score(Y_test, Y_pred)
        # accuracies['Gradient Descent'] = accuracy* 100.0
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    def get_accuracy(self):
        return round(self.accuracy * 100.0, 2)

    def predict_text(self, sentence):
        sentence = preprocess_sentence(sentence)
        train = self.vectorizer.transform([sentence])
        return self.sgd.predict(train)


class LogisticRegressionClass:
    def __init__(self, X_train, X_test, Y_train, Y_test, vectorizer):
        self.vectorizer = vectorizer
        self.logreg = LogisticRegression()
        self.logreg.fit(X_train, Y_train)

        Y_pred = self.logreg.predict(X_test)

        # evaluate predictions
        self.accuracy = accuracy_score(Y_test, Y_pred)
        # accuracies['Logistic Regression'] = accuracy* 100.0
        # print("Accuracy: %.2f%%" % (accuracy * 100.0))

    def get_accuracy(self):
        return round(self.accuracy * 100.0, 2)

    def predict_text(self, sentence):
        sentence = preprocess_sentence(sentence)
        train = self.vectorizer.transform([sentence])
        return self.logreg.predict(train)


class KNN:
    def __init__(self, X_train, X_test, Y_train, Y_test, vectorizer):
        self.vectorizer = vectorizer
        scoreList = []
        for i in range(1, 20):
            self.knn2 = KNeighborsClassifier(
                n_neighbors=i)  # n_neighbors means k
            self.knn2.fit(X_train, Y_train)
            scoreList.append(self.knn2.score(X_test, Y_test))

        # plt.plot(range(1,20), scoreList)
        # plt.xticks(np.arange(1,20,1))
        # plt.xlabel("K value")
        # plt.ylabel("Score")
        # plt.show()

        self.accuracy = max(scoreList)

    def get_accuracy(self):
        return round(self.accuracy * 100.0, 2)

    def predict_text(self, sentence):
        sentence = preprocess_sentence(sentence)
        train = self.vectorizer.transform([sentence])
        return self.knn2.predict(train)
