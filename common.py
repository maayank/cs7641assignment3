import pandas as pd
import numpy as np
from load_datasets import *
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

SCORING = 'f1'

class Experiment:
    '''
    Responsible for preprocessing (e.g. dividing data to sets) and call estimators
    '''
    def __init__(self, scale, grid_search, estimator, interesting_parameters):
        self.estimator = estimator
        self.estimator_name = self.estimator.__class__.__name__
        self.scale = scale
        self.grid_search = grid_search
        if scale:
            self.estimator = make_pipeline(StandardScaler(), self.estimator)
            prefix = self.estimator.steps[1][0]
            interesting_parameters = {f'{prefix}__{k}': v for k, v in interesting_parameters.items()}
        self.interesting_parameters = interesting_parameters

    def load(self, data_name, df):
        np.random.seed(42)
        self.data_name = data_name
        self.training_df, self.test_df = train_test_split(df, train_size=.8, shuffle=True, random_state=42)
        self.training_X, self.training_y = self._split(self.training_df)
        self.test_X, self.test_y = self._split(self.test_df)

    @staticmethod
    def _split(df):
        return df.iloc[:, :-1], df.iloc[:, -1] # last column is the label

    def _eval(self, df):
        test_features, test_y = self._split(df)
        start = perf_counter()
        pred_y = self.estimator.predict(test_features)
        took = perf_counter() - start
 #       print(f'Predicting took {took} seconds.')

        return tuple(func(test_y, pred_y) for func in [accuracy_score, f1_score, precision_score, recall_score])

    def _save_fig(self, name):
        from plot import save_fig
        name = f'{self.data_name}_{self.estimator_name}_{name}'
        save_fig(name)

    def _make_learning_curve(self):
        np.random.seed(42)
        plt.clf()
        from sklearn.model_selection import learning_curve
        train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(self.estimator, self.training_X, self.training_y, cv=5, n_jobs=-1, return_times=True, scoring=SCORING, train_sizes=np.linspace(0.1, 1.0, 5))

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)
        score_times_mean = np.mean(score_times, axis=1)
        score_times_std = np.std(score_times, axis=1)

        _, axes = plt.subplots(1, 2, figsize=(20, 5))
        axes[0].set_ylim(0.7, 1.01)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")
        axes[0].set_title("Learning curve")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-', label="Fit time")
        axes[1].plot(train_sizes, score_times_mean, 'o-', label="Score time")
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].fill_between(train_sizes, score_times_mean - score_times_std,
                             score_times_mean + score_times_std, alpha=0.1)

        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("Time (sec)")
        axes[1].legend(loc="best")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        # axes[2].grid()
        # axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
        #                      test_scores_mean + test_scores_std, alpha=0.1)
        # axes[2].set_xlabel("fit_times")
        # axes[2].set_ylabel("Score")
        # axes[2].set_title("Performance of the model")
        self._save_fig('learning_curve')

    def _make_validation_curve(self):
        np.random.seed(42)
        from sklearn.model_selection import validation_curve
        plt.clf()
        n = len(self.interesting_parameters)
#        n = int(n ** .5)
#        if (n ** 2) < len(self.interesting_parameters):
#            n += 1
        fig, axes = plt.subplots(1, n, figsize=(20,3))
#        fig.suptitle(f"Validation Curves with {self.estimator_name}")
        for i, param_name in enumerate(self.interesting_parameters):
            param_range = self.interesting_parameters[param_name]
            train_scores, test_scores = validation_curve(self.estimator, self.training_X, self.training_y,
            param_name=param_name, param_range=param_range, n_jobs=-1, scoring=SCORING)

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            if all([type(p) is tuple for p in param_range]):
                param_range = list(map(str, param_range))

            if all([type(p) is str for p in param_range]):
#                print(f"Mapping {param_range} for {param_name}")
                param_labels = param_range
                param_range = list(range(len(param_range)))
            else:
                param_labels = None

            if n > 1:
                ax = axes[i]
            else:
                ax = axes
            ax.grid()
            ax.set_xticks(param_range)
            if param_labels is not None:
                ax.set_xticklabels(labels=param_labels)
            ax.set_xlabel(param_name)
            ax.set_ylabel("Score")
            ax.set_ylim(0.0, 1.1)
            lw = 2

            ax.plot(param_range, train_scores_mean, label="Training score",
                         color="darkorange", lw=lw)
            ax.fill_between(param_range, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.2,
                             color="darkorange", lw=lw)
            ax.plot(param_range, test_scores_mean, label="Cross-validation score",
                         color="navy", lw=lw)
            ax.fill_between(param_range, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.2,
                             color="navy", lw=lw)
            ax.legend(loc="best")
        self._save_fig('validation_curves')

    def _grid_search(self):
        np.random.seed(42)
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import classification_report
        clf = GridSearchCV(self.estimator, self.interesting_parameters, n_jobs=-1, scoring=SCORING)
        clf.fit(self.training_X, self.training_y)

#        print(f"Best parameters set found on development set (dataset={self.data_name}, estimator={self.estimator_name}, scale={self.scale}):")
#        print()
#        print(clf.best_params_)
#        print()
#        print("Grid scores on development set:")
#        print()
#        means = clf.cv_results_['mean_test_score']
#        stds = clf.cv_results_['std_test_score']
#        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#            print("%0.3f (+/-%0.03f) for %r"
#                  % (mean, std * 2, params))
#        print()

#        print("Detailed classification report:")
#        print()
#        print("The model is trained on the full development set.")
#        print("The scores are computed on the full evaluation set.")
#        print()
#        y_true, y_pred = self.test_y, clf.predict(self.test_X)
#        print(classification_report(y_true, y_pred))
#        print()
        self.estimator = clf.best_estimator_

    def _make_confusion_matrix(self):
        return
        np.random.seed(42)
        plt.clf()
        from sklearn.metrics import plot_confusion_matrix
        fig = plot_confusion_matrix(self.estimator, self.test_X, self.test_y)
        self._save_fig("confusion_matrix")

    def do(self):
        # First, we do a grid search over the interesting_parameters
        # Then, we print a validation curve on them
        # Then, we choose the best hyperparameters and print a learning curve for them

        np.random.seed(42)

        if self.grid_search:
            self._make_validation_curve()
            self._grid_search()
        else:
            self.estimator.fit(*self._split(self.training_df))

        self._make_learning_curve()
        self._make_confusion_matrix()

#        return

#        start = perf_counter()
#        self.estimator.fit(*self._split(self.training_df))
        #took = perf_counter() - start
#        print(f'Fitting took {took} seconds.')


#        print('Predicting on training_df')
#        print(self._eval(self.training_df))
#        print('----------------')

#        print('Predicting on test_df')
        res = self._eval(self.test_df)
#        print(res)
        return res
