'''
Plan: take the cancer dataset and do k means, PCA and analyze. From there I'll generalize
'''

import pandas as pd
import numpy as np
from load_datasets import *
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, davies_bouldin_score, v_measure_score
from sklearn.decomposition import PCA, FastICA, LatentDirichletAllocation
from plot import *
from scipy.stats import kurtosis
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif
from scipy.linalg import pinv
from copy import deepcopy
from common import Experiment
from sklearn.neural_network import MLPClassifier

np.random.seed(42)

datasets = {
    'cancer': load_cancer(),
    'wine': load_wine()
}

K_MAX = 20
SCATTER = True
RANDOMS = 5

STAGES = list(range(1,6))

cluster_algs = {
    'KMeans': lambda n: KMeans(n_clusters=n),
    'GMM': lambda n: GaussianMixture(n_components=n)
}

# Load data
for name in datasets:
    df = datasets[name]
    df = df.sample(frac=1).reset_index(drop=True)
    X = df.iloc[:, :-1] # drop label
    X_original = X
    X = StandardScaler().fit_transform(X)
    y = df.iloc[:, -1]
    datasets[name] = (X, y, X_original)

def perform_clustering(name, X, tY, scatter, return_stats = False, ret_y=False):
    ret = {}
    for c_name, ctype in cluster_algs.items():
        ss = []
        scores = []
        bic = []
        db = []
        best_mapping_score = []
        ret[c_name] = {}
        for k in range(2, K_MAX):
            clst = ctype(k)
            y = clst.fit_predict(X)
            if ret_y:
                ret[c_name][k] = y
            ss.append(silhouette_score(X, y))
            scores.append(clst.score(X))
            db.append(davies_bouldin_score(X, y))
            best_mapping_score.append(v_measure_score(tY.to_numpy(dtype=int), y, beta=0))
            if 'bic' in dir(clst):
                bic.append(clst.bic(X))
            if scatter and k < 6:
                plot_cluster(name, c_name, X, y, k, tY, weights=clst.score_samples(X) if bic else None)
        if bic:
            stats = {'Log Likelihood': scores, 'Silhouette score': ss, 'DB score': db, 'V measure': best_mapping_score}
        else:
            sse = np.asarray(scores)
            sse *= -1
            sse /= sse.max()
            stats = {'Sum of Squared Errors': sse, 'Silhouette score': ss, 'DB score': db, 'V measure': best_mapping_score}
        if return_stats:
            ret[c_name] = deepcopy(stats)
        elif ret_y:
            pass
        else:
            plot_cluster_stats(name, c_name, stats)
    return ret

# Part 1 - clustering
if 1 in STAGES:
    for name in datasets:
        X, tY, _ = datasets[name]
        perform_clustering(name, X, tY, scatter=SCATTER)

def make_re(X, newX, prj):
    if 'inverse_transform' in dir(prj):
        projX = prj.inverse_transform(newX)
    else:
        # RP
        mat = rp.components_
        projX = (pinv(mat) @ newX.T).T
    prj_err_sq = np.square(X-projX)
    return np.mean(prj_err_sq)


# Part 2 - DR
if 2 in STAGES:
    for name in datasets:
        X, tY, X_original = datasets[name]
        rec_map = {}

        # PCA
        rec_map['PCA'] = []
        pca = PCA()
        newX = pca.fit_transform(X)
        plot_simple(name, 'pca_exp_variance', 'PCA explained variance', '# of features', 'Cumulative exp. variance', pca.explained_variance_ratio_.cumsum())
        mean_kurt = []
        for k in range(1, X.shape[1]+1):
            pca = PCA(n_components=k)
            newX = pca.fit_transform(X)
            rec_map['PCA'].append(make_re(X, newX, pca))
            val = np.abs(kurtosis(newX)).mean()
            mean_kurt.append(val)
        plot_simple(name, 'pca_kurt', 'Kurtosis (PCA)', '# of dimensions', 'Mean abs. kurt.', mean_kurt)

        # ICA
        rec_map['ICA'] = []
        mean_kurt = []
        vanilla_kurt = np.abs(kurtosis(X)).mean()
        print(f'Vanilla kurtosis of {name} is {vanilla_kurt}')
        for k in range(1, X.shape[1]+1):
            ica = FastICA(n_components=k)
            newX = ica.fit_transform(X)
            rec_map['ICA'].append(make_re(X, newX, ica))
            val = np.abs(kurtosis(newX)).mean()
            mean_kurt.append(val)
        plot_simple(name, 'ica_kurt', 'Kurtosis (ICA)', '# of dimensions', 'Mean abs. kurt.', mean_kurt)

        # RP
        rp_rec = []
        for i in range(1, RANDOMS+1):
            curr_rp_rec = []
            for k in range(1, X.shape[1]+1):
                rp = GaussianRandomProjection(n_components=k)
                newX = rp.fit_transform(X)
                curr_rp_rec.append(make_re(X, newX, rp))
            rp_rec.append(curr_rp_rec)
        rp_rec = np.asarray(rp_rec)
        rp_rec = (rp_rec.mean(axis=0), rp_rec.std(axis=0))

        # Select K best mut info
        rec_map['MI'] = []
        mi = SelectKBest(mutual_info_classif, k=X.shape[1])
        newX = mi.fit_transform(X, tY)
        plot_simple(name, 'mi', 'Mututal information', 'Feature (increasing order)', 'Mutual info.', sorted(mi.scores_))
        for k in range(1, X.shape[1]+1):
            mi = SelectKBest(mutual_info_classif, k=k)
            newX = mi.fit_transform(X, tY)
            rec_map['MI'].append(make_re(X, newX, mi))
        plot_rec_map(name, rec_map, rp_rec)

laterXs = {}
chosen_ds = 'cancer'
laterY = None
if 3 in STAGES or 4 in STAGES:
    for name in datasets:
        X, tY, _ = datasets[name]
        k = int(X.shape[1]/2)
        if name == chosen_ds:
            if 'NA' not in laterXs:
                laterXs['NA'] = X
            laterY = tY
        stats = {k:{} for k in cluster_algs}
        for dr_name, dr in {
            'PCA': PCA(k),
            'ICA': FastICA(k),
            'RP': GaussianRandomProjection(k),
            'MI': SelectKBest(mutual_info_classif, k=k)}.items():
            exp_name = f'{name}/{k}/{dr_name}'
            newX = dr.fit_transform(X, tY)
            if name == chosen_ds:
                laterXs[dr_name] = newX
            if 3 in STAGES:
                curr_stats = perform_clustering(exp_name, newX, tY, scatter=False, return_stats=True)
                for c in curr_stats:
                    stats[c][dr_name] = curr_stats[c]
        if 3 not in STAGES:
            assert name == chosen_ds
            break
        for c_name in cluster_algs:
            plot_cluster_stats_batch(f'{name}_batch_{k}', c_name, stats[c_name])


if 4 in STAGES:
    assert laterXs
    tY = laterY
    results = []
    hls_set = [(j,) * i for j in [5, 10, 50] for i in range(1,4)]
    for dr_name, X in laterXs.items():
        experiment = {
                        'estimator': MLPClassifier(learning_rate_init=.001, max_iter=10000),
                        'interesting_parameters': {
                            'hidden_layer_sizes': hls_set
                        }
                    }
        ex = Experiment(0, 1, **experiment)
        df = pd.DataFrame(X)
        df['last'] = tY.astype(int)
        ex.load(f'{chosen_ds}_{dr_name}' , df)
        accuracy, f1, precision, recall = ex.do()
        results.append({
            'DimRed': dr_name,
            'Accuracy': accuracy,
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        })
    print('Stage 4:')
    print(pd.DataFrame(results))
    print('------')

if 5 in STAGES:
    X, tY, _ = datasets['cancer']
    clusters = perform_clustering('stage5cancer', X, tY, scatter=False, ret_y=True)
    hls_set = [(j,) * i for j in [5, 10, 50] for i in range(1,4)]
    results = []
    for c_name in clusters:
        for k in clusters[c_name]:
            X = clusters[c_name][k]
            experiment = {
                        'estimator': MLPClassifier(learning_rate_init=.001, max_iter=10000),
                        'interesting_parameters': {
                            'hidden_layer_sizes': hls_set
                        }
                    }
            ex = Experiment(1, 1, **experiment)
            df = pd.DataFrame(X)
            df['last'] = tY.astype(int)
            ex.load(f'stage5/{c_name}/{k}' , df)
            accuracy, f1, precision, recall = ex.do()
            results.append({
                'Clust.Alg.': c_name,
                'k': k,
                'Accuracy': accuracy,
                'F1': f1,
                'Precision': precision,
                'Recall': recall
            })
    print('Stage 5')
    print(pd.DataFrame(results))
    print('------')

