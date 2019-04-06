import math
import numpy as np
from numpy import isscalar

import sklearn.datasets
import sklearn.model_selection
import sklearn.svm

import optuna
import cma
import rbfopt

from matplotlib import pyplot as plt

simple_log = []


def rosen(x, alpha=1e2):
    """Rosenbrock test objective function"""
    x = [x] if isscalar(x[0]) else x  # scalar into list
    x = np.asarray(x)
    f = [sum(alpha * (x[:-1] ** 2 - x[1:]) ** 2 + (1. - x[:-1]) ** 2) for x in x]
    return f if len(f) > 1 else f[0]  # 1-element-list into scalar


def iris_evaluation(param):
    svc_c = math.pow(10, param[0])
    svc_gamma = math.pow(10, param[1])

    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma=svc_gamma)
    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1, cv=5)
    accuracy = score.mean()
    simple_log.append([param, 1.0 - accuracy])
    print(param, 1.0 - accuracy)
    return 1.0 - accuracy


def objective(trial):
    param = [trial.suggest_uniform('log_svc_c', -5, 5),
             trial.suggest_uniform('log_svc_gamma', -5, 5)]

    score = iris_evaluation(param)
    return score


def plot_simple_log(history):
    p0 = [item[0][0] for item in history]
    p1 = [item[0][1] for item in history]
    value = [item[1] for item in history]
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.scatter(p0, p1, s=10, c=value, cmap='jet')
    plt.colorbar()

    plt.show()


def tune_with_optuna():
    global simple_log
    simple_log = []
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print(study.best_trial)
    history = simple_log
    plot_simple_log(history)
    return history


def tune_with_cmaes():
    global simple_log
    simple_log = []
    es = cma.CMAEvolutionStrategy(2 * [0], 0.5)
    es.logger.disp_header()
    es.optimize(iris_evaluation, iterations=20)  # ``objective_fct``: f(x: array_like) -> float
    print(es.result_pretty())
    history = simple_log
    plot_simple_log(history)
    return history

def tune_with_rbfopt():
    global simple_log
    simple_log = []
    settings = rbfopt.RbfoptSettings(minlp_solver_path='\\Programs\\AMPL\\bonmin-win64\\bonmin.exe',
                                     nlp_solver_path='\\Programs\\AMPL\\ipopt-win64\\ipopt.exe',
                                     max_evaluations=50)
    bb = rbfopt.RbfoptUserBlackBox(2, np.array([-5] * 2), np.array([+5] * 2),
                                   np.array(['R', 'R']), iris_evaluation)
    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    val, x, itercount, evalcount, fast_evalcount = alg.optimize()
    history = simple_log
    plot_simple_log(history)
    return history


if __name__ == '__main__':
    rbfopt_hist = tune_with_rbfopt()
    cma_es_hist = tune_with_cmaes()
    optuna_hist = tune_with_optuna()

    value_rbfopt = [item[1] for item in rbfopt_hist]
    value_cma_es = [item[1] for item in cma_es_hist]
    value_optuna = [item[1] for item in optuna_hist]

    optimal_rbfopt = [np.min(value_rbfopt[0:i + 1]) for i in range(len(value_rbfopt))]
    optimal_cma_es = [np.min(value_cma_es[0:i+1]) for i in range(len(value_cma_es))]
    optimal_optuna = [np.min(value_optuna[0:i+1]) for i in range(len(value_optuna))]

    plt.clf()
    plt.plot(optimal_rbfopt)
    plt.plot(optimal_cma_es)
    plt.plot(optimal_optuna)
    ax = plt.gca()
    ax.set_yscale('log')
    plt.show()

