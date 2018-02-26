
import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize

from downloading import generate_x_y_alexnet

import torch.nn as nn
import time

# a list of class names
from caffe_classes import class_names

# We modify the torchvision implementation so that the features
# after the final pooling layer is easily accessible by calling
#       net.features(...)
# If you would like to use other layer features, you will need to
# make similar modifications.
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)
        
        features_weight_i = [0, 3, 6, 8]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias
            
        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # self.mynet = nn.Sequential(
        #     nn.Dropout(),
        #     torch.nn.Linear(256 * 6 * 6, 12),
        #     nn.ReLU(inplace=True),
        #     torch.nn.Linear(12, 6),
        # )
        self.mynet = nn.Sequential(
            nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            torch.nn.Linear(2048, 12),
            nn.ReLU(inplace=True),
            torch.nn.Linear(12, 6),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self.load_weights()

    def forward(self, x):
        # print("x0: ", x.shape)
        x = self.features(x)
        # print("x1: ", x.shape)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.mynet(x)
        # print("x2: ", x.shape)
        # x = x.view(x.size(0), 256 * 6 * 6)
        # x = self.classifier(x)
        return x

# model_orig = torchvision.models.alexnet(pretrained=True)
# model = MyAlexNet()
# model.eval()
# print(model.features[8].weight.shape)
# print(model.features[8].weight[0].shape)



def train(n):
    torch.manual_seed(1)

    # dtype_float = torch.FloatTensor
    # dtype_long = torch.LongTensor

    dtype_float = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor

    train_x, train_y = generate_x_y_alexnet("training")
    #print(train_x.shape)
    #print(train_y.shape)

    x = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
    y_labels = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)

    myANet = MyAlexNet()
    myANet.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()

    ## TRAINING THE MODEL
    alpha = 1e-2
    # optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    optimizer = torch.optim.SGD(myANet.parameters(), lr=alpha, momentum=0.99)

    for t in range(n):
        if t % 50 == 0:
            print('t: ', t)

        # forward
        y_pred = myANet.forward(x)

        # print('y_labels: ', y_labels)
        # print('y_pred: ', y_pred)
        loss = loss_fn(y_pred, y_labels)
        # print("[Current Loss] ", loss)

        myANet.zero_grad()

        # backward
        loss.backward()
        optimizer.step()
        

    print('==========TRAINING SET==========')
    y_pred = myANet(x).data.cpu().numpy()
    # print(y_pred.shape)
    # print(train_y.shape)
    training_performance = np.mean(np.argmax(y_pred, 1) == np.argmax(train_y, 1))
    print("[Performance - TRAINING SET] ", (training_performance * 100), "%\n")

    print('==========Validation SET==========')

    validating_x, validating_y = generate_x_y_alexnet("validating")

    x = Variable(torch.from_numpy(validating_x), requires_grad=False).type(dtype_float)
    y_pred = myANet(x).data.cpu().numpy()

    validating_performance = np.mean(np.argmax(y_pred, 1) == np.argmax(validating_y, 1))
    print("[Performance - Validation SET] ", (validating_performance * 100), "%\n")

    print('==========TEST SET==========')

    test_x, test_y = generate_x_y_alexnet("testing")

    x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)
    y_pred = myANet(x).data.cpu().numpy()

    test_performance = np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))
    print("[Performance - TEST SET] ", (test_performance * 100), "%\n")

    # print("[np.argmax(y_pred, 1)] ", np.argmax(y_pred, 0))

    # print("[np.argmax(test_y, 0))] ", np.argmax(test_y, 1))

    # print("(np.argmax(y_pred, 1) == np.argmax(test_y, 0))", (np.argmax(y_pred, 1) == np.argmax(test_y, 0)))

    # print("[y_pred[5]] ", y_pred[5])

    # print("[test_y.T[5]] ", test_y.T[5])

    # print("[model[0].weight] ", model[0].weight)

    # print("[model[0].weight.data.numpy()[0, :].shape] ", model[0].weight.data.numpy()[0, :].shape)

    # plt.imshow(model[0].weight.data.numpy()[0, :].reshape((32, 32)), cmap=plt.cm.coolwarm)

    # plt.imshow(model[0].weight.data.numpy()[1, :].reshape((32, 32)), cmap=plt.cm.coolwarm)
    # plt.show()

    return training_performance, validating_performance, test_performance, myANet


if __name__ == "__main__":
    ## Get Learning Curve

    training_performances = list()
    validating_performances = list()
    test_performances = list()

    max_iter = 250
    addon = list(range(50, max_iter, 50))
    iters = [1, 2, 5]
    iters.extend(addon)

    for i in iters:
        print("========================= Current Iteration: {} =========================\n".format(i))
        print("==================== General Parameters ====================")
        print("[Maximum # of Iterations] ", max_iter, '\n')
        print("==================== Current Information =====================")
        training_performance, validating_performance, test_performance, model = train(i)
        training_performances.append(training_performance)
        validating_performances.append(validating_performance)
        test_performances.append(test_performance)
        # torch.cuda.empty_cache()
        print("=========================================================================\n")

    print("==================== Results ====================")
    print("[training_performances] ", training_performances)
    print("[validating_performances] ", validating_performances)
    print("[test_performances] ", test_performances)
    print("[Best # of Iterations] ", iters[np.argmax(test_performances)])
    print("[Best test_performance] ", (test_performances[np.argmax(test_performances)] * 100), "%")

    plot_x = iters
    plt.plot(plot_x, training_performances, 'c', label='Training Set Performances')
    plt.plot(plot_x, validating_performances, 'y', label='Validation Set Performances')
    plt.plot(plot_x, test_performances, 'm', label='Test Set Performances')
    plt.legend(loc='best')
    plt.xlabel("Number of Iterations", fontsize=10.5)
    plt.ylabel("Performances (%)", fontsize=10.5)
    plt.savefig('part10-performances-plot.png')
    plt.gcf().clear()
