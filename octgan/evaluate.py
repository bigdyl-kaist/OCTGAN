import json
import numpy as np
import pandas as pd
from pomegranate import BayesianNetwork
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, silhouette_score, matthews_corrcoef
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

from octgan.constants import CATEGORICAL, CONTINUOUS, ORDINAL

_MODELS = {
    'binary_classification': [
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
                'max_depth': 20
            }
        },
        {
            'class': AdaBoostClassifier,
        },
        {
            'class': LogisticRegression,
            'kwargs': {
                'solver': 'lbfgs',
                'n_jobs': -1,
                'max_iter': 50
            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (50, ),
                'max_iter': 50
            },
        }
    ],
    'multiclass_classification': [
        {
            'class': DecisionTreeClassifier,
            'kwargs': {
                'max_depth': 30,
                'class_weight': 'balanced',
            }
        },
        {
            'class': MLPClassifier,
            'kwargs': {
                'hidden_layer_sizes': (100, ),
                'max_iter': 50
            },
        }
    ],
    'regression': [
        {
            'class': LinearRegression,
        },
        {
            'class': MLPRegressor,
            'kwargs': {
                'hidden_layer_sizes': (100, ),
                'max_iter': 50
            },
        }
    ],

    'clustering': [
        {
            'class': KMeans, 
            'kwargs': {
                'n_clusters': 2,
                'n_jobs': -1,
            }
        }
    ]
}


class FeatureMaker:

    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):

        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]

        features = []
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    labels = col.astype(int)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col, 1e-2))

                else:
                    feature = (col - cmin) / (cmax - cmin) 

            elif cinfo['type'] == ORDINAL:
                feature = col

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features, labels


def _prepare_ml_problem(train, test, metadata, clustering=False): 
    fm = FeatureMaker(metadata)
    x_train, y_train = fm.make_features(train)
    x_test, y_test = fm.make_features(test)
    if clustering:
        model = _MODELS["clustering"]
    else:
        model = _MODELS[metadata['problem_type']]
    return x_train, y_train, x_test, y_test, model


def _evaluate_multi_classification(train, test, metadata):
   
    """Score classifiers using f1 score and the given train and test data.

    Args:
        x_train(numpy.ndarray):
        y_train(numpy.ndarray):
        x_test(numpy.ndarray):
        y_test(numpy):
        classifiers(list):

    Returns:
        pandas.DataFrame
    """
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)

    performance = []
    f1 = [] 
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)

        report = classification_report(y_test, pred, output_dict=True)
        classes = list(report.keys())[:-3]
        proportion = [  report[i]['support'] / len(y_test) for i in classes]
        weighted_f1 = np.sum(list(map(lambda i, prop: report[i]['f1-score']* (1-prop)/(len(classes)-1), classes, proportion)))
                
        f1.append([report[c]['f1-score'] for c in classes] )
        acc = accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average='macro')
        micro_f1 = f1_score(y_test, pred, average='micro')

        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                'weighted_f1': weighted_f1,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1
            }
        )

    return pd.DataFrame(performance)


def _evaluate_binary_classification(train, test, metadata):
   
    x_train, y_train, x_test, y_test, classifiers = _prepare_ml_problem(train, test, metadata)
    performance = []
    f1 = [] 
    for model_spec in classifiers:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        unique_labels = np.unique(y_train)
        if len(unique_labels) == 1:
            pred = [unique_labels[0]] * len(x_test)
            pred_prob = np.array([1.] * len(x_test))

        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred_prob = model.predict_proba(x_test)


        acc = accuracy_score(y_test, pred)
        binary_f1 = f1_score(y_test, pred, average='binary')
        macro_f1 = f1_score(y_test, pred, average='macro')
        report = classification_report(y_test, pred, output_dict=True)
        classes = list(report.keys())[:-3]

        f1.append([report[c]['f1-score'] for c in classes] )

        mcc = matthews_corrcoef(y_test, pred)

        precision = precision_score(y_test, pred, average='binary')
        recall = recall_score(y_test, pred, average='binary')
        size = [a["size"] for a in metadata["columns"] if a["name"] == "label"][0]
        rest_label = set(range(size)) - set(unique_labels)
        tmp = []
        j = 0
        for i in range(size):
            if i in rest_label:
                tmp.append(np.array([0] * y_test.shape[0])[:,np.newaxis])
            else:
                try:
                    tmp.append(pred_prob[:,[j]])
                except:
                    tmp.append(pred_prob[:, np.newaxis])
                j += 1
        roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))

        performance.append(
            {
                "name": model_repr,
                "accuracy": acc,
                "binary_f1": binary_f1,
                "macro_f1": macro_f1,
                "matthews_corrcoef": mcc, 
                "precision": precision,
                "recall": recall,
                "roc_auc": roc_auc
            }
        )
    
    return pd.DataFrame(performance)

def _evaluate_regression(train, test, metadata):
   
    x_train, y_train, x_test, y_test, regressors = _prepare_ml_problem(train, test, metadata)

    performance = []
    y_train = np.log(np.clip(y_train, 1, 20000))
    y_test = np.log(np.clip(y_test, 1, 20000))
    for model_spec in regressors:
        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__
        model = model_class(**model_kwargs)

        model.fit(x_train, y_train)
        pred = model.predict(x_test)

        r2 = r2_score(y_test, pred)
        explained_variance = explained_variance_score(y_test, pred)
        mean_squared = mean_squared_error(y_test, pred)
        mean_absolute = mean_absolute_error(y_test, pred)



        performance.append(
            {
                "name": model_repr,
                "r2": r2,
                "explained_variance" : explained_variance,
                "mean_squared_error" : mean_squared,
                "mean_absolute_error" : mean_absolute
            }
        )

    return pd.DataFrame(performance)

def _evaluate_gmm_likelihood(train, test, metadata, components=[10, 30]):
    results = list()
    for n_components in components:
        gmm = GaussianMixture(n_components, covariance_type='diag')
        gmm.fit(test)
        l1 = gmm.score(train)

        gmm.fit(train)
        l2 = gmm.score(test)

        results.append({
            "name": repr(gmm),
            "syn_likelihood": l1,
            "test_likelihood": l2,
        })

    return pd.DataFrame(results)

def _mapper(data, metadata):
    data_t = []
    for row in data:
        row_t = []
        for id_, info in enumerate(metadata['columns']):
            row_t.append(info['i2s'][int(row[id_])])

        data_t.append(row_t)

    return data_t

def _evaluate_bayesian_likelihood(train, test, metadata):
    structure_json = json.dumps(metadata['structure'])
    bn1 = BayesianNetwork.from_json(structure_json)

    train_mapped = _mapper(train, metadata)
    test_mapped = _mapper(test, metadata)
    prob = []
    for item in train_mapped:
        try:
            prob.append(bn1.probability(item))
        except Exception:
            prob.append(1e-8)

    l1 = np.mean(np.log(np.asarray(prob) + 1e-8))

    bn2 = BayesianNetwork.from_structure(train_mapped, bn1.structure)
    prob = []

    for item in test_mapped:
        try:
            prob.append(bn2.probability(item))
        except Exception:
            prob.append(1e-8)

    l2 = np.mean(np.log(np.asarray(prob) + 1e-8))

    return pd.DataFrame([{
        "name": "Bayesian Likelihood",
        "syn_likelihood": l1,
        "test_likelihood": l2,
    }])


def _evaluate_cluster(train, test, metadata):
   
    x_train, y_train, x_test, y_test, kmeans = _prepare_ml_problem(train, test, metadata, clustering=True)
 

    model_class = kmeans[0]['class']
    model_repr = model_class.__name__
    unique_labels = np.unique(y_train)
    num_columns = metadata['columns'][-1]["size"]
    
    result = []
    for i in range(3):
        model = model_class(n_clusters = num_columns*(i+1))

        if len(unique_labels) == 1:
            result.append([unique_labels[0]] * len(x_test))

        else:
            try:
                model.fit(x_train)
                predicted_label = model.predict(x_test)
            except:
                x_train = x_train.astype(np.float32)
                model.fit(x_train)

                x_test = x_test.astype(np.float32)
                predicted_label = model.predict(x_test)
            try:
                result.append(silhouette_score(x_test, predicted_label, metric='euclidean', sample_size=100))
            except:
                result.append(0)
        

    return pd.DataFrame([{
        "name": model_repr,
        "silhouette_score": np.mean(result),
    }])



_EVALUATORS = {
    'bayesian_likelihood': [_evaluate_bayesian_likelihood],
    'gaussian_likelihood': [_evaluate_gmm_likelihood],
    'regression': [_evaluate_regression],
    'binary_classification': [_evaluate_binary_classification, _evaluate_cluster],
    'multiclass_classification': [_evaluate_multi_classification, _evaluate_cluster]
}

def compute_scores(test, synthesized_data, metadata):
    result = pd.DataFrame()

    for evaluator in _EVALUATORS[metadata['problem_type']]:
        scores = pd.DataFrame()
        
        for i in range(5):
            score = evaluator(synthesized_data, test, metadata) 
            score['test_iter'] = i
            scores = pd.concat([scores, score], ignore_index=True)
        scores = scores.groupby(['test_iter']).mean() 
        result = pd.concat([result, scores], axis=1)

    return result
