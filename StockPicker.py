from sklearn import svm
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def createEqualAmountOfClassesHardcoded(data):
    klasse1 = data[data['Class'] == 1]
    klasse2 = data[data['Class'] == 0]

    klasse1 = klasse1.iloc[0:1346, :]

    frames = [klasse1, klasse2]

    result = pd.concat(frames)

    return result

def fillNonExistentValuesScaleDataAndSelectMostRelevantFeatures(X, data):
    X = X.fillna(data.mean())

    X = MinMaxScaler().fit_transform(X)

    m = SelectFromModel(svm.SVC(max_iter=100000, C=30, kernel='linear', probability=True))

    m.fit(X, y)

    X = m.transform(X)

    return X

def trainClassifier(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    classifier = svm.SVC(max_iter=100000, C=30, kernel='linear', probability=True)
    # classifier = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(10, 10, 10, 5), random_state=1)
    classifier.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, classifier

def pickOutWinnersBasedOnProbabilityFromClassifier(cutoffpoint, Chosenclassifier, Xvalue):
    dataOld = pd.read_csv("2018_Financial_Data.csv")

    #classifier = Chosenclassifier

    probabilities = Chosenclassifier.predict_proba(Xvalue)

    df = pd.DataFrame(probabilities, columns=['Column_A', 'Column_B'])
    cutoff = cutoffpoint
    df = df.loc[(df['Column_B'] >= cutoff)]
    listWithIndex = df.index.values.tolist()

    winners = 0
    loosers = 0
    for entry in listWithIndex:
        stock = dataOld.iloc[entry, [224]]
        if stock['Class'] == 1:
            winners = winners+1
        else:
            loosers = loosers+1
    print('Percentage of correct winners:')
    print(winners/(winners+loosers))

    print('number of stocks:')
    print(len(listWithIndex))

    print('stock names:')
    print(dataOld.iloc[listWithIndex, [0, 224]])

def trainClassifierAndPickOutBestStocks(cutoffpoint, Xval, yval, data):
    X = fillNonExistentValuesScaleDataAndSelectMostRelevantFeatures(Xval, data)

    X_train, X_test, y_train, y_test, classifier = trainClassifier(X, yval)

    Y_pred = classifier.predict(X_test)

    pickOutWinnersBasedOnProbabilityFromClassifier(cutoffpoint, classifier, X)

    print('score of classifier:')
    print(classifier.score(X, y))

    F1 = f1_score(y_test, Y_pred, average='micro')

    acc = accuracy_score(y_test, Y_pred)
    print('accuracy score:')
    print(acc)

    print('f1 score: ')
    print(F1)

data = pd.read_csv("2018_Financial_Data.csv")
y = data['Class']
X = data.iloc[:, 1:221]

trainClassifierAndPickOutBestStocks(0.85, X, y, data)






