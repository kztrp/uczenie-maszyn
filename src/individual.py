import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier

class Individual:
    def __init__(self, n_features, data):
        self.genotype_features = np.zeros((3, n_features))
        self.genotype_algorithms = np.zeros(3)
        self.scores = np.zeros(3)
        self.fitness_scores = np.zeros(5)
        self.data = data
        self.iteration = 0
        self.tag = 0

    def __str__(self) -> str:
        representation = ""
        representation += "{}\n".format(self.genotype_features)
        representation += "{}\n".format(self.genotype_algorithms)
        representation += "Created in {} iteration\n".format(self.iteration)
        representation += "Score: {}\n".format(self.scores)
        representation += "Fitness: {}\n".format(self.fitness_scores)
        return representation


    def __repr__(self):
        return f"Individual({self.genotype_features}, {self.genotype_algorithms}\
            , {self.scores}, {self.fitness_scores}, {self.iteration})"

    def generate_genotype(self):
        self.genotype_features = np.random.randint(0, 2, self.genotype_features.shape)
        self.genotype_algorithms = np.random.randint(0, 2, self.genotype_algorithms.shape)
        while np.nonzero(self.genotype_features)[0].size == 0:
            self.genotype_features = np.random.randint(0, 2, self.genotype_features.shape)
        while np.nonzero(self.genotype_algorithms)[0].size == 0:
            self.genotype_algorithms = np.random.randint(0, 2, self.genotype_algorithms.shape)
        self.calculate_score()

    def calculate_score(self):
        clfs = (GaussianNB(), KNeighborsClassifier(n_neighbors=7), LinearDiscriminantAnalysis())

        for i, alg in enumerate(self.genotype_algorithms):
            if alg == 1:
                features = np.nonzero(self.genotype_features[i])[0]
                X = self.data[:, features]
                y = self.data[:, -1]
                sc = StandardScaler()
                X = sc.fit_transform(X)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
                scores = []
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    clfs[i].fit(X_train, y_train)
                    predict = clfs[i].predict(X_test)
                    scores.append(accuracy_score(y_test, predict))
                self.scores[i] = np.mean(scores)
        if np.nonzero(self.scores)[0].size > 0:
            self.fitness_scores[0] = np.sum(self.scores, where=self.genotype_algorithms.astype('bool'))
            self.fitness_scores[1] = np.prod(self.scores, where=self.genotype_algorithms.astype('bool'))
            self.fitness_scores[2] = np.max(self.scores[np.nonzero(self.scores)])
            self.fitness_scores[3] = np.min(self.scores[np.nonzero(self.scores)])
            self.fitness_scores[4] = np.median(self.scores[np.nonzero(self.scores)])


    def breeding(self, ind_a, ind_b, mutation_rate, split_points):
        for j in range(3):
            for i in range(len(self.genotype_features[j])):
                if i < split_points[0]:
                    self.genotype_features[j][i] = ind_a.genotype_features[j][i]
                else:
                    self.genotype_features[j][i] = ind_b.genotype_features[j][i]
        for i in range(len(self.genotype_algorithms)):
            if i < split_points[1]:
                self.genotype_algorithms[i] = ind_a.genotype_algorithms[i]
            else:
                self.genotype_algorithms[i] = ind_b.genotype_algorithms[i]
        self.mutate(mutation_rate)
        for i in range(3):
            if np.nonzero(self.genotype_features[i])[0].size == 0:
                r = random.randint(0, self.genotype_features[i].size-1)
                self.genotype_features[i][r] = 1
        self.calculate_score()

    def mutate(self, mutation_rate):
        for j in range(3):
            for i in range(len(self.genotype_features[j])):
                if random.uniform(0, 1) <  mutation_rate:
                    self.genotype_features[j][i] = random.randint(0, 1)
        for i in range(len(self.genotype_algorithms)):
            if random.uniform(0, 1) <  mutation_rate:
                self.genotype_algorithms[i] = random.randint(0, 1)

    def calculate_combined(self):
        clfs = (GaussianNB(), KNeighborsClassifier(n_neighbors=7), LinearDiscriminantAnalysis())
        features = np.nonzero(self.genotype_features[0])[0]
        X = self.data[:, features]
        y = self.data[:, -1]
        sc = StandardScaler()
        X = sc.fit_transform(X)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1234)
        clf = VotingClassifier(estimators=[('nb', clfs[0]), ('knn', clfs[1]), ('lda', clfs[2])])
        scores = []
        for train_index, test_index in skf.split(X, y):
            probs = np.zeros((3, test_index.size, 2))
            probs.fill(0.5)
            for i in range(3):
                if self.genotype_algorithms[i] == 1:
                    features = np.nonzero(self.genotype_features[i])[0]
                    X = self.data[:, features]
                    y = self.data[:, -1]
                    sc = StandardScaler()
                    X = sc.fit_transform(X)
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    clfs[i].fit(X_train, y_train)
                    probs[i] = clfs[i].predict_proba(X_test)
            scores.append(accuracy_score(y_test, self.combine(probs)))
        print(np.mean(scores))

    def combine(self, probs):
        pred2 = np.average(probs, axis=0)
        pred = np.argmax(pred2, axis=1)
        pred += 1
        return pred
