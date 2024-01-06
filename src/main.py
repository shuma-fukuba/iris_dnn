import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from iris_dnn import config as c
from iris_dnn.dataset import load_iris
from iris_dnn.DNN import DNN

torch.manual_seed(c.SEED)


class Parser:
    def __init__(self, epochs=c.EPOCH) -> None:
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
        self.train_X = train_X
        self.val_X = val_X
        self.train_y = train_y
        self.val_y = val_y

        self.model = DNN()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.02)

        self.epochs = epochs

    def train(self):
        train_loss = []
        train_acc = []

        self.model.train()
        for epoch in range(self.epochs):
            data, target = Variable(self.train_X), Variable(self.train_y)
            self.optimizer.zero_grad()
            output = self.model(data)

            loss = F.nll_loss(output, target)
            loss.backward()
            train_loss.append(loss.data.item())
            self.optimizer.step()

            prediction = output.data.max(1)[1]
            acc = prediction.eq(target.data).sum().numpy() / len(self.train_X)
            train_acc.append(acc)

            if epoch % 10 == 0:
                print(
                    f'Train Step: {epoch} Loss: {loss.data.item()} Accuracy: {acc}')
            epoch += 1

        print('Train Step: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(
            epoch, loss.data.item(), acc))

    def eval(self):
        self.model.eval()
        outputs = self.model(Variable(self.val_X))
        _, predicted = torch.max(outputs.data, 1)
        print(
            f'Acc: {predicted.eq(self.val_y).sum().numpy() / len(predicted)}')


if __name__ == "__main__":
    parser = Parser()
    parser.train()
    print(parser.model)
    parser.eval()
