from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import _tree
import sys

def model_size(rf):
    def tree_size(tree, curNode=0):
        if tree.children_left[curNode] == _tree.TREE_LEAF and tree.children_right[curNode] == _tree.TREE_LEAF:
            return 1
        else:
            size_left = tree_size(tree, tree.children_left[curNode])
            size_right = tree_size(tree, tree.children_right[curNode])
            return 1 + size_left + size_right

    total_size = None
    for e in rf.estimators_:
        if isinstance(e, RandomForestClassifier):
            for t in e.estimators_:
                if total_size is None:
                    total_size = tree_size(t.tree_, 0)
                else:
                    total_size += tree_size(t.tree_, 0)
        else:
            if total_size is None:
                total_size = tree_size(e.tree_, 0)
            else:
                total_size += tree_size(e.tree_, 0)
    return total_size

def type_guess(str):
    import ast
    try:
        return ast.literal_eval(str)
    except:
        try:
            return class_guess(str)
        except:
            return str

def class_guess(classname):
    return getattr(sys.modules[__name__], classname)

def consume_extra_arguments(extra_args, fun):
    import inspect
    args = inspect.signature(fun).parameters
    ret = {}
    for a in args:
        if f"--{a}" in extra_args:
            ind = extra_args.index(f"--{a}")
            val = type_guess(extra_args[ind + 1])
            ret[a] = val 
            extra_args.pop(ind+1)
            extra_args.pop(ind)
    print(f"{fun.__name__}() consumes the following parameters:", ret)
    return ret
