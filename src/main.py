from individual import Individual
from genetic_algorithm import GeneticAlgorithm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


def main():
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    encoder = LabelEncoder()
    df = pd.read_csv('indian_liver.csv')
    data = df.to_numpy()
    data[:, 1] = encoder.fit_transform(data[:, 1])
    data = imr.fit_transform(data)
    alg = GeneticAlgorithm(data)
    alg.generate_population(30, df.shape[1]-1)
    alg.run_algorithm(50, 1)
if __name__ == '__main__':
    main()
