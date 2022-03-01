"""
This files originate from the "New-Empty-Python-Project-Base" template:
    https://github.com/Neuraxio/New-Empty-Python-Project-Base 
Created by Guillaume Chevalier at Neuraxio:
    https://github.com/Neuraxio 
    https://github.com/guillaume-chevalier 
License: CC0-1.0 (Public Domain)
"""


def distill(forest, max_depth):
    from proxy_model import ProxyModel
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np
    from sklearn.tree import _tree
    proxy_model = ProxyModel(forest, max_distillation_depth=max_depth)
    proxy_model.fit()

    tree = DecisionTreeClassifier()
    # tree.tree_
