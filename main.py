from preprocess import get_data, preprocess
from models import RandomForest, SGD, LogisticRegressionClass, KNN
import pandas as pd
import tkinter as tk
X_train, X_test, Y_train, Y_test, vectorizer = get_data()

root = tk.Tk()
root.minsize(200, 400)
root.title("Personality Predictor")
btn_list = []

rf = RandomForest(X_train, X_test, Y_train, Y_test, vectorizer)
sgd = SGD(X_train, X_test, Y_train, Y_test, vectorizer)
lr = LogisticRegressionClass(
    X_train, X_test, Y_train, Y_test, vectorizer)
knn = KNN(X_train, X_test, Y_train, Y_test, vectorizer)


def onClick(index):
    print(index)  # Print the index value
    # Print the text for the selected button
    print(btn_list[index].cget("text"))
    if btn_list[index].cget("text") == "RandomForest":
        message = rf.get_accuracy()
        label = tk.Label(root, text=message, bg="light green",
                         font=('Helvetica 20 bold'))
        label.grid(row=index, column=1)
    elif btn_list[index].cget("text") == "SGD":
        message = sgd.get_accuracy()
        label = tk.Label(root, text=message, bg="light green",
                         font=('Helvetica 20 bold'))
        label.grid(row=index, column=1)
    elif btn_list[index].cget("text") == "LogisticRegression":
        message = lr.get_accuracy()
        label = tk.Label(root, text=message, bg="light green",
                         font=('Helvetica 20 bold'))
        label.grid(row=index, column=1)
    elif btn_list[index].cget("text") == "KNN":
        message = knn.get_accuracy()
        label = tk.Label(root, text=message, bg="light green",
                         font=('Helvetica 20 bold'))
        label.grid(row=index, column=1)


def predictOnClick(model):

    user_input = inputtxt.get(1.0, "end-1c")

    # if model == 'sgd':
    #     predicted = sgd.predict_text(user_input)
    #     print('-------------', predicted[0])
    #     lbl.config(text="Predicted Personality: "+predicted[0])

    predicted = model.predict_text(user_input)
    lbl.config(text="Predicted Personality: "+predicted[0])


for i in range(4):
    # Lambda command to hold reference to the index matched with range value
    if i == 0:
        b = tk.Button(root, text='RandomForest',
                      command=lambda idx=i: onClick(idx))
    elif i == 1:
        b = tk.Button(root, text='SGD', command=lambda idx=i: onClick(idx))
    elif i == 2:
        b = tk.Button(root, text='LogisticRegression',
                      command=lambda idx=i: onClick(idx))
    elif i == 3:
        b = tk.Button(root, text='KNN', command=lambda idx=i: onClick(idx))
    b.grid(row=i, column=0)
    btn_list.append(b)  # Append the button to a list

inputtxt = tk.Text(root, height=5, width=20)
inputtxt.grid(row=6, column=0)

rfBtn = tk.Button(root, text='Random Forest predict', command=lambda : predictOnClick(rf))
rfBtn.grid(row=7, column=0)

sgdBtn = tk.Button(root, text='SGD predict', command=lambda : predictOnClick(sgd))
sgdBtn.grid(row=7, column=1)

lrBtn = tk.Button(root, text='LogisticRegression predict', command=lambda : predictOnClick(lr))
lrBtn.grid(row=7, column=2)

knnBtn = tk.Button(root, text='KNN predict', command=lambda : predictOnClick(knn))
knnBtn.grid(row=7, column=3)

lbl = tk.Label(root, text="")
lbl.grid(row=8, column=0)
root.mainloop()

#######################################################################

# rf = RandomForest(X_train, X_test, Y_train, Y_test, vectorizer)
# print(rf.predict_text("I'm finding the lack of me in these posts very alarming.Giving new meaning to 'Game' theory.|||Hello *ENTP Grin*  That's all it takes.."))

# sgd = SGD(X_train, X_test, Y_train, Y_test, vectorizer)
# print(sgd.predict_text("I'm finding the lack of me in these posts very alarming.|||Sex can be boring if it's in the same position often. "))

# logreg = LogisticRegressionClass(X_train, X_test, Y_train, Y_test, vectorizer)
# print(logreg.predict_text("I'm finding the lack of me in these posts very alarming.|||Sex can be boring if it's in the same position often. "))

# knn = KNN(X_train, X_test, Y_train, Y_test, vectorizer)
# print(knn.predict_text("I'm finding the lack of me in these posts very alarming.|||Sex can be boring if it's in the same position often. "))
