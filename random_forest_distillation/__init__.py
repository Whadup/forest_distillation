"""
This files originate from the "New-Empty-Python-Project-Base" template:
    https://github.com/Neuraxio/New-Empty-Python-Project-Base 
Created by Guillaume Chevalier at Neuraxio:
    https://github.com/Neuraxio 
    https://github.com/guillaume-chevalier 
License: CC0-1.0 (Public Domain)
"""


def distill(forest, max_depth=None, max_nodes=None, return_forest=False):
    from queue import Queue
    from .proxy_model import ProxyModel
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    from sklearn.tree import _tree as internal_tree
    proxy_model = ProxyModel(forest, max_distillation_depth=max_depth, max_nodes=max_nodes)
    proxy_model.fit()

    # tree.feature,
    # tree.threshold,
    # tree.children_left,
    # tree.children_right,
    # 1.0 * tree.value[:, 0, 1] / tree.value[:, 0, :].sum(axis=-1)
    Q = Queue()
    Q.put(proxy_model.root_node)
    feature = [0 for _ in range(proxy_model.num_nodes)]
    threshold = [0 for _ in range(proxy_model.num_nodes)]
    children_left = [0 for _ in range(proxy_model.num_nodes)]
    children_right = [0 for _ in range(proxy_model.num_nodes)]
    value = [0 for _ in range(proxy_model.num_nodes)]
    impurity = []
    n_node_samples = []
    weighted_n_node_samples = []
    # print(proxy_model.root_node.output)
    while not Q.empty():
        n = Q.get()
        value[n.node_id]= (n.output)
        impurity.append(0.0)
        n_node_samples.append(1)
        weighted_n_node_samples.append(1.0)
        if n.left != -1:
            feature[n.node_id]= (n.feature)
            threshold[n.node_id]= n.threshold
            children_left[n.node_id]= (n.left.node_id)
            children_right[n.node_id]= (n.right.node_id)
            Q.put(n.left)
            Q.put(n.right)
        else:
            threshold[n.node_id]= -2
            feature[n.node_id] = (-2)
            children_left[n.node_id]= (-1)
            children_right[n.node_id]= (-1)
    
    bla = internal_tree.Tree(proxy_model.n_features, np.array([2]), 1)
    v1 = np.array(value).reshape(-1,1) * 10000000
    v2 = 10000000 - np.array(value).reshape(-1,1) * 10000000
    
    values = np.stack((v2, v1), axis=2).astype(int).astype(float)

    state = {
        'n_features_': proxy_model.n_features,
        'max_depth': proxy_model.max_depth,
        'node_count': proxy_model.num_nodes,
        'nodes': np.array([tuple(x) for x in zip(children_left, children_right, feature, threshold, impurity, n_node_samples, weighted_n_node_samples)],
                          dtype=[('left_child', '<i8'), ('right_child', '<i8'), ('feature', '<i8'),('threshold', '<f8'), ('impurity', '<f8'), ('n_node_samples', '<i8'), ('weighted_n_node_samples', '<f8')]),
        'values': values
    }
    bla.__setstate__(state)
        
    new_tree = DecisionTreeClassifier()
    new_tree.tree_ = bla
    new_tree.n_outputs_ = 1
    new_tree.n_classes_ = 2
    if return_forest:
        new_forest = RandomForestClassifier(n_estimators=1)
        new_forest.estimators_ = [new_tree]
        new_forest.n_classes_ = 2
        new_forest.n_outputs_ = 1
        new_forest.n_features_in_ = forest.n_features_in_
        new_forest.classes_ = forest.classes_
        return new_forest
    return new_tree

