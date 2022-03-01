"""
Proxy regression tree model for approximating a given random forest classifier
"""
import numpy as np
from copy import deepcopy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
from sklearn.base import BaseEstimator, ClassifierMixin

from .kernels import inner_loop, intervals_intersect, variance, skewness, kurtosis, bias
import tqdm
import os
from queue import PriorityQueue, Queue, Empty
import multiprocessing
from time import sleep
# import threading
LEAF = _tree.TREE_LEAF
ZERO = 1e-14
FRESH = 0
DONE = 10
# Multiprocessing workaround as we only need self in split node for some hyperparameters.
SELF_ = None
DEBUG = (0.6165042812900634,4.560736427077957e-08, 2.8117135331291076e-08, 1.7522512857487604e-08 ,1.1029905992249491e-08  ,7.007403354011706e-09)

def setup(s):
    global SELF_
    SELF_ = s

def set_X_y(X, y):
    global SELF_
    SELF_.X = X
    SELF_.y0 = y
    return 1

def split_node(item):
    from datetime import datetime
    global SELF_
    # print("split", datetime.now())
    return SELF_.split_node(item)
# End Multiprocessing workaround

def volume(theta):
    return (theta[..., 1] - theta[..., 0]).prod(-1).clip(0, None)

def overlap(theta1, theta2):
    l, r = np.maximum(theta1[..., 0], theta2[..., 0]), np.minimum(theta1[..., 1], theta2[..., 1])
    return np.stack((l, np.maximum(l, r)), -1)

def volume_overlap(theta1, theta2):
    l, r = np.maximum(theta1[..., 0], theta2[..., 0]), np.minimum(theta1[..., 1], theta2[..., 1])
    return (np.maximum(l, r) - l).prod(-1)

def absmax_(a, b):
    if np.abs(a) > np.abs(b):
        return a
    return b

absmax = np.frompyfunc(absmax_, 2, 1)

class Node():
    node_id = 0
    depth = 0
    status = FRESH
    left = LEAF
    right = LEAF
    output = 0.5
    feature = None
    threshold = None
    theta = None
    loss = np.inf
    def __hash__(self):
        return self.node_id
    def __eq__(self, other):
        if isinstance(other, int):
            return False
        return self.node_id == other.node_id
    def __lt__(self, other):
        return False

class ProxyModel():
    """
    n_estimators = number of trees
    n_features = data dimension
    trees = [(features, values, lefts, rights, outputs)]
    """
    def __init__(self, tree_ensemble, max_distillation_depth=4):
        self.max_depth = max_distillation_depth
        if isinstance(tree_ensemble, RandomForestClassifier):
            self.n_estimators = tree_ensemble.n_estimators
            self.n_features = tree_ensemble.n_features_in_
            self.original_prediction = tree_ensemble
            self.trees = []
            for tree in tree_ensemble:
                tree = tree.tree_
                assert (0.0 <= tree.threshold[tree.children_left != LEAF]).all() and (tree.threshold[tree.children_left != LEAF] <= 1.0).all(), \
                    "We only support distillation when all features x are in 0 <= x <= 1"
                self.trees.append(
                    (
                        tree.feature,
                        tree.threshold,
                        tree.children_left,
                        tree.children_right,
                        1.0 * tree.value[:, 0, 1] / tree.value[:, 0, :].sum(axis=-1)
                    )
                )
        elif isinstance(tree_ensemble, list):
            self.n_estimators = len(tree_ensemble)
            self.n_features = max(tree[0].max() for tree in tree_ensemble) + 1
            self.trees = tree_ensemble
        else:
            raise RuntimeError("Currently we support RandomForestClassifiers and manually providing the trees")
        self.root_node = None

    def overlap_query(self, tree, query_theta, search_theta=None, node=0, drop_feature=None):
        if search_theta is None:
            search_theta = np.array([[0.0, 1.0] for i in range(self.n_features)])
        features, thresholds, lefts, rights, outputs = tree
        feature, threshold, left, right, output = features[node], thresholds[node], lefts[node], rights[node], outputs[node]
        if left == LEAF and right == LEAF:
            if drop_feature is None:
                v = volume(overlap(query_theta, search_theta))
            else:
                mask = np.ones(query_theta[..., 0].shape, dtype=bool)
                mask[drop_feature] = False
                v = volume(overlap(query_theta[mask], search_theta[mask]))
            return 2 * v * output
        agg = 0.0
        #check if left child overlaps with query_theta
        if intervals_intersect(*query_theta[feature], search_theta[feature][0], threshold):
            tmp = search_theta[feature][1]
            search_theta[feature][1] = threshold
            agg += self.overlap_query(tree, query_theta, search_theta, left)
            search_theta[feature][1] = tmp
        #check if right child overlaps with query_theta
        if intervals_intersect(*query_theta[feature], threshold, search_theta[feature][1]):
            tmp = search_theta[feature][0]
            search_theta[feature][0] = threshold
            agg += self.overlap_query(tree, query_theta, search_theta, right)
            search_theta[feature][0] = tmp
        return agg

    def overlap_query_construction(self, tree, query_theta, search_theta=None, node=0):
        if search_theta is None:
            search_theta = np.array([[0.0, 1.0] for i in range(self.n_features)])
        features, thresholds, lefts, rights, outputs = tree
        feature, threshold, left, right, output = features[node], thresholds[node], lefts[node], rights[node], outputs[node]
        if left == LEAF and right == LEAF:
            ov = overlap(query_theta, search_theta)
            return [(2 * output / self.n_estimators, ov)] if volume(ov) > 0 else []
        agg = []
        #check if left child overlaps with query_theta
        if intervals_intersect(*query_theta[feature], search_theta[feature][0], threshold):
            tmp = search_theta[feature][1]
            search_theta[feature][1] = threshold
            agg += self.overlap_query_construction(tree, query_theta, search_theta, left)
            search_theta[feature][1] = tmp
        #check if right child overlaps with query_theta
        if intervals_intersect(*query_theta[feature], threshold, search_theta[feature][1]):
            tmp = search_theta[feature][0]
            search_theta[feature][0] = threshold
            agg += self.overlap_query_construction(tree, query_theta, search_theta, right)
            search_theta[feature][0] = tmp
        return agg

    def b(self, theta, drop_feature=None):
        agg = 0
        for tree in self.trees:
            agg += self.overlap_query(tree, theta, drop_feature=drop_feature)
        return agg
    
    def b_construction(self, theta):
        agg = []
        for tree in self.trees:
            agg += self.overlap_query_construction(tree, theta)
        return agg

    def c(self, theta, drop_feature=None):
        if drop_feature is None:
            return volume(theta)
        else:
            mask = np.ones(theta[..., 0].shape, dtype=bool)
            mask[drop_feature] = False
            return volume(theta[mask])
    
    def a(self, node):
        return variance(node)

    def split_node(self, node):
        #Find a split for node
        best_split_feauture = None
        best_split_value = None
        best_split_gain = -np.inf
        best_split_output = None
        best_split_b = None

        # for split_feature in np.random.choice(np.arange(self.n_features), size=min(2, int(np.sqrt(self.n_features))), replace=False):
        for split_feature in range(self.n_features):
            base_c = self.c(node.theta, drop_feature=split_feature)
            mask = np.ones(self.n_features, dtype=bool)
            mask[split_feature] = False
            base_b = [(v * volume(t[mask]), t[split_feature]) for v, t in zip(node.partitions_value, node.partitions_theta)] #TODO: Make this two numpy arrays?
            theta_k = node.theta[split_feature] #valid range of the k-th feature in the current node

            ids = np.argsort(node.partitions_theta[:, split_feature, :].reshape(-1))
            split_values = np.ascontiguousarray(node.partitions_theta[:, split_feature, :].reshape(-1)[ids])
            split_partitions = np.ascontiguousarray(ids // 2)

            #TODO: Handle nominal attributes more efficiently...

            b_l = 0
            b_r = sum([v * (t[1] - t[0]) for v, t in base_b])

            last_split_value = theta_k[0]
            values = np.array([b[0] for b in base_b])
            starts = np.array([b[1][0] for b in base_b])

            gain, split_value, split_output, split_b = inner_loop(b_l,
                b_r,
                base_c,
                split_values,
                split_partitions,
                last_split_value,
                values,
                starts,
                theta_k[0], theta_k[1], best_split_gain)
            if gain > best_split_gain:
                best_split_gain = gain
                best_split_feauture = split_feature
                best_split_value = split_value
                best_split_output = split_output
                best_split_b = split_b


        #Construct two children, lot's of book-keeping going on...
        node.feature = best_split_feauture
        node.threshold = best_split_value
        node.left = Node()
        node.left.node_id = node.node_id * 2
        node.left.depth = node.depth + 1
        node.left.theta = deepcopy(node.theta)
        node.left.theta[node.feature, 1] = best_split_value
        node.left.output = best_split_output[0]
        node.left.partitions_theta = deepcopy(node.partitions_theta)
        node.left.partitions_theta[:, best_split_feauture, 1] = np.minimum(node.left.partitions_theta[:, best_split_feauture, 1], best_split_value)
        still_active = node.left.partitions_theta[:, best_split_feauture, 1] > node.left.partitions_theta[:, best_split_feauture, 0]
        node.left.partitions_theta = node.left.partitions_theta[still_active]
        node.left.partitions_value = node.partitions_value[still_active]
        v  = volume(node.left.theta)
        a = 0
        if node.left.depth == self.max_depth:
            a = self.a(node.left)
            node.left.loss = 0.0 if v < ZERO else a - best_split_b[0] ** 2 / (4 * v)

        t = np.abs(node.left.output - 0.5) #/ np.sqrt(node.left.loss)
        node.left.danger_zone = v if t < ZERO else min(t**(-2) * node.left.loss, v)

        
    

        node.right = Node()
        node.right.node_id = node.node_id * 2 + 1
        node.right.depth = node.depth + 1
        node.right.theta = deepcopy(node.theta)
        node.right.theta[node.feature, 0] = best_split_value
        node.right.output = best_split_output[1]
        node.right.partitions_theta = deepcopy(node.partitions_theta)
        node.right.partitions_theta[:, best_split_feauture, 0] = np.maximum(node.right.partitions_theta[:, best_split_feauture, 0], best_split_value)
        still_active = node.right.partitions_theta[:, best_split_feauture, 1] > node.right.partitions_theta[:, best_split_feauture, 0]
        node.right.partitions_theta = node.right.partitions_theta[still_active]
        node.right.partitions_value = node.partitions_value[still_active]
        v  = volume(node.right.theta)
        if node.right.depth == self.max_depth:
            node.right.loss = 0.0 if v < ZERO else self.a(node.right) - best_split_b[1] ** 2 / (4 * v)

        t = np.abs(node.right.output - 0.5) #/ np.sqrt(node.right.loss)
        node.right.danger_zone = v if t < ZERO else min(t**(-2) * node.right.loss, v)# 
        
        return 1, node, node.left, node.right

    def fit(self, *args):

        Q = Queue()
    
        root_theta = np.array([[0.0, 1.0] for i in range(self.n_features)])
        self.root_node = Node()
        self.root_node.loss = np.inf
        self.root_node.theta = root_theta

        partitions = self.b_construction(root_theta)
        self.root_node.partitions_theta = np.array([y for x, y in partitions])
        self.root_node.partitions_value = np.array([x for x, y in partitions])
        # print(self.root_node.output, bias(self.root_node), 2 * self.c(root_theta))
        self.root_node.output = bias(self.root_node) / (2 * self.c(root_theta))
        Q.put(self.root_node)
        c = 0
        danger_zone = 0
        
        node_dict = {self.root_node.node_id:self.root_node}

        for i in tqdm.tqdm(range(2**(self.max_depth) - 1), total=2**(self.max_depth) - 1):
            x = Q.get()
            _, n, l, r = self.split_node(x)
            tmp = node_dict[n.node_id]

            tmp.left = l
            tmp.right = r
            tmp.threshold = n.threshold
            tmp.feature = n.feature
            tmp.output = n.output
            tmp.loss = n.loss

            if l != LEAF:
                l.node_id = len(node_dict)
                node_dict[l.node_id] = l
                Q.put(l)
                r.node_id = len(node_dict)
                node_dict[r.node_id] = r
                Q.put(r)

        loss = 0
        num_nodes = 0
        max_depth = 0
        Q = Queue()
        Q.put(self.root_node)
        while not Q.empty():
            node = Q.get()
            num_nodes += 1
            max_depth = max(max_depth, node.depth)
            if node.left == LEAF:
                loss += node.loss
            else:
                Q.put(node.left)
                Q.put(node.right)
        print("num_nodes", num_nodes, "max_depth", max_depth)
        self.loss = loss
        self.num_nodes = num_nodes
        self.max_depth = max_depth

        
    def predict_proba(self, X):
        y = []
        for x in X:
            node = self.root_node
            while True:
                if node.left == LEAF and node.right == LEAF:
                    y.append(node.output)
                    break
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
        return np.array(y)

