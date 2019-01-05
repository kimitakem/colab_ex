import sklearn.datasets
import sklearn.model_selection
import sklearn.svm


def objective(trial):
    iris = sklearn.datasets.load_iris()
    x, y = iris.data, iris.target

    svc_c = trial.suggest_loguniform('svc_c', 1e-10, 1e10)
    svc_gamma = trial.suggest_loguniform('svc_gamma', 1e-10, 1e10)
    classifier_obj = sklearn.svm.SVC(C=svc_c, gamma=svc_gamma)

    score = sklearn.model_selection.cross_val_score(classifier_obj, x, y, n_jobs=-1)
    accuracy = score.mean()
    print('svc_c', svc_c)
    print('svc_gamma', svc_gamma)
    print('score', 1.0 - accuracy)

    return 1.0 - accuracy

def tune_with_optuna():
    import optuna
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    df_trial = study.trials_dataframe()

    from matplotlib import pyplot as plt

    plt.xscale('log')
    plt.yscale('log')
    plt.scatter(df_trial.params.svc_c, df_trial.params.svc_gamma,
                s=100, c=df_trial.value, cmap='jet')
    plt.colorbar()
    print(study.best_trial)

    plt.show()

if __name__ == '__main__':
    import cma
    es = cma.CMAEvolutionStrategy(8 * [0], 0.5)
    es.optimize(cma.ff.rosen)
    print(es.result_pretty())

