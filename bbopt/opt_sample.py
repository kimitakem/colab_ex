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
import lightgbm as lgb

simple_log = []




def gbm_eval(param):
    num_leaves = int(math.pow(10, param[0]))
    learning_rate = math.pow(10, param[1])
    data, target = sklearn.datasets.load_breast_cancer(return_X_y=True)
    train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(data, target, test_size=0.25)
    dtrain = lgb.Dataset(train_x, label=train_y)
    gbm_param = {'objective': 'binary',
                 'metric': 'binary_logloss',
                 'verbosity': -1,
                 'boosting_type': 'gbdt',
                 'num_leaves': num_leaves,
                 'learning_rate': learning_rate
    }
    gbm = lgb.train(gbm_param, dtrain)
    preds = gbm.predict(test_x)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
    simple_log.append([param, 1.0 - accuracy])
    print(param, 1.0 - accuracy)
    return 1.0 - accuracy


def rosen_eval(param, alpha=1e2):
    """Rosenbrock test objective function"""
    x = [param] if isscalar(param[0]) else param  # scalar into list
    x = np.asarray(x)
    f = [sum(alpha * (x[:-1] ** 2 - x[1:]) ** 2 + (1. - x[:-1]) ** 2) for x in x]
    f_value = f if len(f) > 1 else f[0]  # 1-element-list into scalar
    simple_log.append([param, f_value])
    print(param, f_value)
    return f_value


def iris_eval(param):
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


def plot_simple_log(history):
    p0 = [item[0][0] for item in history]
    p1 = [item[0][1] for item in history]
    value = [item[1] for item in history]
    #plt.xlim(-5, 5)
    #plt.ylim(-5, 5)
    plt.scatter(p0, p1, s=10, c=value, cmap='jet')
    plt.colorbar()

    plt.show()


def tune_with_optuna(fc_name):
    global simple_log
    simple_log = []
    study = optuna.create_study()

    def rosen_objective(trial):
        param = [trial.suggest_uniform('x_0', -5, 5),
                 trial.suggest_uniform('x_1', -5, 5),
                 trial.suggest_uniform('x_2', -5, 5),
                 trial.suggest_uniform('x_3', -5, 5),
                 trial.suggest_uniform('x_4', -5, 5),
                 trial.suggest_uniform('x_5', -5, 5),
                 trial.suggest_uniform('x_6', -5, 5),
                 trial.suggest_uniform('x_7', -5, 5)
                 ]
        score = rosen_eval(param)
        return score

    def iris_objective(trial):
        param = [trial.suggest_uniform('log_svc_c', -5, 5),
                 trial.suggest_uniform('log_svc_gamma', -5, 5)]

        score = iris_eval(param)
        return score

    def lgb_objective(trial):
        param = [trial.suggest_uniform('log_num_leaves', 1, 3),
                 trial.suggest_uniform('log_learning_rate', -8, 0)
                 ]
        return gbm_eval(param)

    if fc_name is 'rosen':
        study.optimize(rosen_objective, n_trials=100)
    elif fc_name is 'iris':
        study.optimize(iris_objective, n_trials=100)
    elif fc_name is 'gbm':
        study.optimize(lgb_objective, n_trials=100)
    else:
        raise NotImplementedError

    print(study.best_trial)
    history = simple_log
    plot_simple_log(history)
    return history


def tune_with_cmaes(obj_name):
    global simple_log
    simple_log = []

    if obj_name is "rosen":
        param_dim = 8
        obj_fc = rosen_eval
        es = cma.CMAEvolutionStrategy(param_dim * [0], 0.5)
    elif obj_name is "iris":
        param_dim = 2
        obj_fc = iris_eval
        es = cma.CMAEvolutionStrategy(param_dim * [0], 0.5)
    elif obj_name is "gbm":
        obj_fc = gbm_eval
        es = cma.CMAEvolutionStrategy([2, -4], 0.5)
    else:
        raise NotImplementedError

    es.logger.disp_header()
    es.optimize(obj_fc, iterations=20)  # ``objective_fct``: f(x: array_like) -> float
    print(es.result_pretty())
    history = simple_log
    plot_simple_log(history)
    return history


def tune_with_rbfopt(obj_name):
    global simple_log
    simple_log = []
    settings = rbfopt.RbfoptSettings(minlp_solver_path='\\Programs\\AMPL\\bonmin-win64\\bonmin.exe',
                                     nlp_solver_path='\\Programs\\AMPL\\ipopt-win64\\ipopt.exe',
                                     max_evaluations=100)

    if obj_name == 'rosen':
        param_dim = 8
        obj_fc = rosen_eval
        bb = rbfopt.RbfoptUserBlackBox(param_dim, np.array([-5] * param_dim), np.array([5] * param_dim),
                                       np.array(['R'] * param_dim), obj_fc)
    elif obj_name == 'iris':
        param_dim = 2
        obj_fc = iris_eval
        bb = rbfopt.RbfoptUserBlackBox(param_dim, np.array([-5] * param_dim), np.array([5] * param_dim),
                                       np.array(['R'] * param_dim), obj_fc)
    elif obj_name == 'gbm':
        bb = rbfopt.RbfoptUserBlackBox(dimension=2,
                                       var_lower=np.array([1, -8]),
                                       var_upper=np.array([3, 0]),
                                       var_type=np.array(['R', 'R']),
                                       obj_funct=gbm_eval)
    else:
        raise NotImplementedError

    alg = rbfopt.RbfoptAlgorithm(settings, bb)
    val, x, itercount, evalcount, fast_evalcount = alg.optimize()

    history = simple_log
    plot_simple_log(history)
    return history


if __name__ == '__main__':
    obj_name = 'gbm'
    rbfopt_hist = tune_with_rbfopt(obj_name)
    cma_es_hist = tune_with_cmaes(obj_name)
    optuna_hist = tune_with_optuna(obj_name)

    value_rbfopt = [item[1] for item in rbfopt_hist]
    value_cma_es = [item[1] for item in cma_es_hist]
    value_optuna = [item[1] for item in optuna_hist]

    optimal_rbfopt = [np.min(value_rbfopt[0:i + 1]) for i in range(len(value_rbfopt))]
    optimal_cma_es = [np.min(value_cma_es[0:i+1]) for i in range(len(value_cma_es))]
    optimal_optuna = [np.min(value_optuna[0:i+1]) for i in range(len(value_optuna))]

    plt.clf()
    plt.plot(optimal_rbfopt, label="rbfopt")
    plt.plot(optimal_cma_es, label="cma_es")
    plt.plot(optimal_optuna, label="optuna")
    plt.legend()
    ax = plt.gca()
    ax.set_yscale('log')
    plt.show()

