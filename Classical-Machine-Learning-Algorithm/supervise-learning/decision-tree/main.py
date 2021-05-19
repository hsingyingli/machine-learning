import pandas as pd 
import random

from model import DecisionTreeClassifier




def main():
    df = pd.read_csv('./data/iris.csv', index_col=0)
    train_idx = random.sample(range(len(df)), int(0.8*len(df)))
    test_idx  = [i for i in range(len(df)) if i not in train_idx]
    train_feature = df.iloc[train_idx, :-1].values
    train_label   = df.iloc[train_idx, -1].values
    test_feature = df.iloc[test_idx, :-1].values
    test_label   = df.iloc[test_idx, -1].values

    dt = DecisionTreeClassifier()
    dt.fit(train_feature, train_label)
    pred = dt.predict(test_feature)
    print(pred)
    print(test_label)




if __name__ == '__main__':
    main()