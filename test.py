# Call for Macbook Pro and Cluster respectively:
# LDFLAGS=-"L/usr/local/opt/llvm/lib" CC=/usr/local/opt/llvm/bin/clang python setup.py build_ext --inplace && python -m random_forest_robustness.robustness
# python setup.py build_ext --inplace && python -m random_forest_robustness.robustness
import argparse
import numpy as np
import pandas as pd
import pickle
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances
from scipy.stats import mode
from random_forest_distillation import distill
from random_forest_distillation.utils import model_size, consume_extra_arguments, type_guess
from meticulous import Experiment

def get_data(dataset="MagicTelescope", quantile_transform=True):
    """Load data, split intro train and test and normalize each feature."""
    X, y = fetch_openml(dataset, return_X_y=True)
    # binary targets majority vs rest
    majority_class, _ = mode(y, axis=None)
    y = (y == majority_class.item()).astype(int)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(pd.get_dummies(X), y, test_size=0.2, stratify=y, random_state=1504)
    if quantile_transform:
        q = make_pipeline(SimpleImputer(strategy="most_frequent"), QuantileTransformer(), )
    else:
        q = make_pipeline(SimpleImputer(strategy="most_frequent"), MinMaxScaler(), )
    X_train = q.fit_transform(X_train)
    X_test = q.transform(X_test)
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    #    "MagicTelescope", "mozilla4", "higgs",  "Adult", "Bank", "Nomao", "electricity", "eeg-eye-state", 
    #    "click_prediction_small" , "PhishingWebsites", "Amazon_employee_access"
    import multiprocessing
    # multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", type=str, default="MagicTelescope")
    parser.add_argument("--model", type=str, default="RandomForestClassifier")
    parser.add_argument("--max_distillation_depth", type=int)
    parser.add_argument("--max_distillation_nodes", type=int)
    Experiment.add_argument_group(parser)
    args, extra_args = parser.parse_known_args()
    #Make extra arguments regular arguments for meticulous tracking...
    for x, y in zip(extra_args[::2], extra_args[1::2]):
        parser.add_argument(x, action="store", type=type(type_guess(y)))
    
    args = parser.parse_args()
    experiment = Experiment.from_parser(parser)

    print(f"========~ {args.dataset} ~========")
    X_train, y_train, X_test, y_test = get_data(args.dataset, **consume_extra_arguments(extra_args, get_data))

    model = type_guess(args.model)
    defaults = dict(n_jobs=-1, random_state=1504)
    model = model(**(defaults | consume_extra_arguments(extra_args, model)))
    print(f"MODEL = {model}")
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)
    print("TEST ACCURACY", model.score(X_test, y_test))
    experiment.summary(test_accuracy=model.score(X_test, y_test))
    print("MODEL SIZE", model_size(model))
    experiment.summary(model_size=model_size(model))

    proxy_model = distill(model, max_depth=args.max_distillation_depth, max_nodes=args.max_distillation_nodes, return_forest=True)
    # print("DISTILLATION LOSS", proxy_model.loss)
    # experiment.summary(proxy_loss=proxy_model.loss)
    proxy_acc = accuracy_score(y_test, proxy_model.predict_proba(X_test).argmax(axis=1))
    print("PROXY ACCURACY", proxy_acc)
    experiment.summary(proxy_accuracy=proxy_acc)


   
    if len(extra_args):
        print("WARNING! Unused command line arguments remaining:", extra_args)
