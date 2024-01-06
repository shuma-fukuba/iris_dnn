import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_dataset() -> tuple[np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray]:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target_names[iris.target]

    y = np.array(df['target'].astype('category').cat.codes).astype(float)
    X = np.array(df.iloc[:, :4])
    train_X, val_X, train_y, val_y = train_test_split(
        X, y, test_size=0.2, random_state=71)

    train_X = torch.Tensor(train_X)
    val_X = torch.Tensor(val_X)
    train_y = torch.LongTensor(train_y)
    val_y = torch.LongTensor(val_y)

    return train_X, val_X, train_y, val_y
