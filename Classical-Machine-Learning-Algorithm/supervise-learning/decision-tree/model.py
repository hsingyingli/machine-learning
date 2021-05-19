import numpy as np 
import pandas as pd
import random 
from collections import namedtuple


class Node(object):
    def __init__(self, feature_idx = None, info_gain=None, threshold = None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value
        self.threshold = threshold


class DecisionTreeClassifier(object):
    def __init__(self, max_depth=2, min_batch=2):
        self.root = None
        self.min_batch = min_batch
        self.max_depth = max_depth
        self.Best = namedtuple('Best', ['info_gain', 'feature_idx', 'threshold', 'left', 'right'])
        self.Pair = namedtuple('Pair', ['feature', 'label'])
    def fit(self, x, y, criterion = 'gini'):
        self.root = self.get_node(x, y, criterion)

    def predict(self, x):
        prediction = [self.single_prediction(feature, self.root) for feature in x]
        return prediction
    def single_prediction(self, feature, tree):
        if tree.value != None:
            return tree.value
        feature_val = feature[tree.feature_idx]
        if feature_val <= tree.threshold:
            return self.single_prediction(feature, tree.left)
        
        return self.single_prediction(feature, tree.right)

    def get_node(self, x, y, criterion, curr_depth = 0):
        x = np.array(x)
        y = np.array(y)
        batch_size, feature_size = x.shape
        dataset = self.Pair(feature = x, label = y)
        if batch_size >= self.min_batch and curr_depth <= self.max_depth:
            best = self.greedy(dataset, batch_size, feature_size, criterion)
            if best.info_gain > 0:
                left_subtree = self.get_node(best.left.feature, best.left.label, criterion, curr_depth+1)
                right_subtree = self.get_node(best.right.feature, best.right.label, criterion, curr_depth+1)
                return Node(best.feature_idx, best.info_gain, best.threshold, left_subtree, right_subtree)
        value = self.get_value(y)
        return Node(value = value)

    def greedy(self, dataset, batch_size, feature_size, criterion):
        best = None
        max_info_gain = -float('inf')
        for idx in range(feature_size):
            feature_value = dataset.feature[:, idx]
            thresholds = np.unique(feature_value)
            for threshold in thresholds:
                left, right = self.split(dataset, idx, threshold)
                if len(left.label)>0 and len(right.label)>0:
                    info_gain = self.information_gain(dataset.label, left.label, right.label, criterion)
                    if info_gain > max_info_gain:
                        best = self.Best(info_gain = info_gain, feature_idx = idx, threshold = threshold, left = left, right = right)
                        
                        max_info_gain = info_gain
        return best

    def split(self, dataset, idx, threshold):
        left_feature  = []
        left_label    = []
        right_feature = []
        right_label   = []
        for row in range(len(dataset.feature)):
            if dataset.feature[row, idx]<= threshold:
                left_feature.append(np.array(dataset.feature[row]))
                left_label.append(np.array(dataset.label[row]))
            else:
                right_feature.append(np.array(dataset.feature[row]))
                right_label.append(np.array(dataset.label[row]))
        return self.Pair(feature = left_feature, label = left_label), self.Pair(feature = right_feature, label = right_label)
    
    def information_gain(self, parent, left_child, right_child, criterion):
        weight_l = len(left_child) / len(parent)
        weight_r = len(right_child) / len(parent)
        info_gain = 0.
        if criterion == 'gini':
            info_gain = self.gini(parent) - (weight_l*self.gini(left_child) + weight_r*self.gini(right_child))
        elif criterion == 'entropy':
            info_gain = self.entropy(parent) - (weight_l*self.entropy(left_child) + weight_r*self.entropy(right_child))
        else:
            raise Exception('criterion error')
        return info_gain

    def gini(self, y):
        classes = np.unique(y)
        gini = 0.
        for cls in classes:
            p = len([i for i in y if i == cls]) / len(y)
            gini += p**2
        return 1-gini

    def entropy(self, y):
        classes = np.unique(y)
        entropy = 0.
        for cls in classes:
            p = len([i for i in y if i == cls]) / len(y)
            entropy += -p*np.log2(p)
        return entropy
    def get_value(self, y):
        y = list(y)
        return max(y, key = y.count)
    



if __name__ == '__main__':
    # df = pd.read_csv('./supervise-learning/decision-tree/data/iris.csv', index_col=0)
    df = pd.read_csv('./data/iris.csv', index_col=0)
    train_idx = random.sample(range(len(df)), int(0.8*len(df)))
    test_idx  = [i for i in range(len(df)) if i not in train_idx]
    train_feature = df.iloc[train_idx, :-1].values
    train_label   = df.iloc[train_idx, -1].values
    test_feature = df.iloc[test_idx, :-1].values
    test_label   = df.iloc[test_idx, -1].values

    print(train_feature.shape)
    print(train_label.shape)
    print(test_feature.shape)
    print(test_label.shape)
    dt = DecisionTreeClassifier()
    dt.fit(train_feature, train_label)
    dt.print_tree()
    pred = dt.predict(test_feature)
    print(pred)
    print(test_label)
