from individual import Individual
from genetic_algorithm import GeneticAlgorithm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler



def main():
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    encoder = LabelEncoder()
    scaler = StandardScaler()
    df = pd.read_csv('indian_liver.csv')
    data = df.to_numpy()
    data[:, 1] = encoder.fit_transform(data[:, 1])
    data = imr.fit_transform(data)
    alg = GeneticAlgorithm(data)
    n_pop = int(input("Wielkość populacji: "))
    alg.generate_population(n_pop, df.shape[1]-1)
    n_it = int(input("Liczba pokoleń: "))
    alg.run_algorithm(n_it, 0)
    print(f"Dokładność klasyfikatora utworzonego z rozwiązania: {alg.population[0].calculate_combined()}")

if __name__ == '__main__':
    main()
