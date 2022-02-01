import numpy as np
from scipy.stats import ttest_ind, ttest_rel
from tabulate import tabulate
from individual import Individual
from genetic_algorithm import GeneticAlgorithm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from simple_classifier import simple_classifiers, combine

def main():
    criterion = 4

    res_gen = test_ga(criterion)
    res_simple = test_simple_classifiers()
    results = np.zeros((2,10))
    results[0] = res_simple
    results[1] = res_gen
    print(results)
    print(f"Podejście tradycyjne, średnia: {np.mean(res_simple):.5f}, wariancja: {np.var(res_simple):.5f}")
    print(f"Algorytm genetyczny, średnia: {np.mean(res_gen):.5f}, wariancja: {np.var(res_gen):.5f}")
    test_classifiers(results)

def test_classifiers(results):
    alfa = .05
    t_statistic = np.zeros((2, 2))
    p_value = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            t_statistic[i, j], p_value[i, j] = ttest_rel(results[i], results[j])
    print("t-statistic:\n", t_statistic, "\n\np-value:\n", p_value)

    headers = ["Prosty", "Genetyczny"]
    names_column = np.array([["Prosty"], ["Genetyczny"]])
    t_statistic_table = np.concatenate((names_column, t_statistic), axis=1)
    t_statistic_table = tabulate(t_statistic_table, headers, floatfmt=".2f")
    p_value_table = np.concatenate((names_column, p_value), axis=1)
    p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
    print("t-statistic:\n", t_statistic_table, "\n\np-value:\n", p_value_table)

    advantage = np.zeros((2, 2))
    advantage[t_statistic > 0] = 1
    advantage_table = tabulate(np.concatenate(
        (names_column, advantage), axis=1), headers)
    print("Advantage:\n", advantage_table)

    significance = np.zeros((2,2))
    significance[p_value <= alfa] = 1
    significance_table = tabulate(np.concatenate(
        (names_column, significance), axis=1), headers)
    print("Statistical significance (alpha = 0.05):\n", significance_table)

    stat_better = significance * advantage
    stat_better_table = tabulate(np.concatenate(
        (names_column, stat_better), axis=1), headers)
    print("Statistically significantly better:\n", stat_better_table)

def test_ga(criterion):
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    encoder = LabelEncoder()
    df = pd.read_csv('indian_liver.csv')
    data = df.to_numpy()
    data[:, 1] = encoder.fit_transform(data[:, 1])
    # data[:, -1] = encoder.fit_transform(data[:, -1])
    data = imr.fit_transform(data)
    results_ga = np.zeros(10)
    for i in range(10):
        alg = GeneticAlgorithm(data)
        alg.generate_population(30, df.shape[1]-1)
        alg.run_algorithm(50, criterion)
        results_ga[i] = alg.population[0].calculate_combined()
    return results_ga

def test_simple_classifiers():
    imr = SimpleImputer(missing_values=np.nan, strategy='mean')
    encoder = LabelEncoder()
    df = pd.read_csv('indian_liver.csv')
    data = df.to_numpy()
    data[:, 1] = encoder.fit_transform(data[:, 1])
    # data[:, -1] = encoder.fit_transform(data[:, -1])
    data = imr.fit_transform(data)
    X = data[:, :-1]
    y = data[:, -1]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    results = np.zeros(10)
    for i in range(10):
        # results[i] = simple_classifiers(X, y)
        results[i] = combine(X, y)
    # print(np.mean(results, axis=0))
    return results

if __name__ == '__main__':
    main()
