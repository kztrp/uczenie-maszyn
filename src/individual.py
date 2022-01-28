import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

class Individual:
    def __init__(self, n_features, data):
        self.genotype_features = np.zeros(n_features)
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
        self.genotype_features = np.random.randint(0, 2, len(self.genotype_features))
        self.genotype_algorithms = np.random.randint(0, 2, len(self.genotype_algorithms))
        while np.nonzero(self.genotype_features)[0].size == 0:
            self.genotype_features = np.random.randint(0, 2, len(self.genotype_features))
        while np.nonzero(self.genotype_algorithms)[0].size == 0:
            self.genotype_algorithms = np.random.randint(0, 2, len(self.genotype_algorithms))
        self.calculate_score()

    def calculate_score(self):
        clfs = (GaussianNB(), KNeighborsClassifier(n_neighbors=7), LinearDiscriminantAnalysis())
        features = np.nonzero(self.genotype_features)[0]
        X = self.data[:, features]
        y = self.data[:, -1]
        sc = StandardScaler()
        X = sc.fit_transform(X)
        for i, alg in enumerate(self.genotype_algorithms):
            if alg == 1:
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
        for i in range(len(self.genotype_features)):
            if i < split_points[0]:
                self.genotype_features[i] = ind_a.genotype_features[i]
            else:
                self.genotype_features[i] = ind_b.genotype_features[i]
        for i in range(len(self.genotype_algorithms)):
            if i < split_points[1]:
                self.genotype_algorithms[i] = ind_a.genotype_algorithms[i]
            else:
                self.genotype_algorithms[i] = ind_b.genotype_algorithms[i]
        self.mutate(mutation_rate)
        if np.nonzero(self.genotype_features)[0].size == 0:
            r = random.randint(0, self.genotype_features.size-1)
            self.genotype_features[r] = 1
        self.calculate_score()

    def mutate(self, mutation_rate):
        for i in range(len(self.genotype_features)):
            if random.uniform(0, 1) <  mutation_rate:
                self.genotype_features[i] = random.randint(0, 1)
        for i in range(len(self.genotype_algorithms)):
            if random.uniform(0, 1) <  mutation_rate:
                self.genotype_algorithms[i] = random.randint(0, 1)