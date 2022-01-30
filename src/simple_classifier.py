from individual import Individual
from genetic_algorithm import GeneticAlgorithm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier


def main():
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    encoder = LabelEncoder()
    df = pd.read_csv('indian_liver.csv')
    data = df.to_numpy()
    data[:, 1] = encoder.fit_transform(data[:, 1])
    data = imr.fit_transform(data)
    X = data[:, :-1]
    y = data[:, -1]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    test_simple_classifiers(X, y)

def test_simple_classifiers(X, y):
    results = np.zeros((10, 5))
    for i in range(10):
        results[i] = simple_classifiers(X, y)
    print(np.mean(results, axis=0))
    return results


def simple_classifiers(X, y):
    significant_features = []
    f_stat, p_value = f_classif(X, y)
    print(p_value)
    mean_scores = np.zeros(3)
    fitness_scores = np.zeros(5)
    for i in range(len(p_value)):
        if p_value[i] >= 0.05:
            significant_features.append(i)
    X = X[:, significant_features]
    clfs = (GaussianNB(), KNeighborsClassifier(n_neighbors=7), LinearDiscriminantAnalysis())
    for i, clf in enumerate(clfs):
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            clfs[i].fit(X_train, y_train)
            predict = clfs[i].predict(X_test)
            scores.append(accuracy_score(y_test, predict))
        mean_scores[i] = np.mean(scores)
    fitness_scores[0] = np.sum(mean_scores)
    fitness_scores[1] = np.prod(mean_scores)
    fitness_scores[2] = np.max(mean_scores)
    fitness_scores[3] = np.min(mean_scores)
    fitness_scores[4] = np.median(mean_scores)
    print(fitness_scores)
    return(fitness_scores)

def combine(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=542)
    clf = VotingClassifier(estimators=[('nb', GaussianNB()), ('knn', KNeighborsClassifier(n_neighbors=7)), ('lda', LinearDiscriminantAnalysis())], voting='soft')
    scores = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)
        scores.append(accuracy_score(y_test, predict))
    print(np.mean(scores))

if __name__ == '__main__':
    main()
