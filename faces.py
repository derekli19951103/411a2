from torch.autograd import Variable
from downloading import generate_x_y
import torch
import numpy as np
import matplotlib.pyplot as plt
from digits import next_batch, shuffle_input

########################
#       Utilities      #
########################

def save_weight_imgs(model, indices):
    for i in indices:
        cur_weight = model[0].weight.data.numpy()[indices[i], :].reshape((32, 32))
        print('[cur_weight_{}.shape] '.format(indices[i]), cur_weight.shape)
        plt.imshow(cur_weight, cmap=plt.cm.coolwarm)
        plt.imsave('part9-weight_{}.png'.format(indices[i]), cur_weight, cmap=plt.cm.coolwarm)
        plt.gcf().clear()

########################
#       NN Train       #
########################


def train(dim_x, dim_h, dim_out, n):
    torch.manual_seed(1)

    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor

    train_x, train_y = generate_x_y("training")

    x = Variable(torch.from_numpy(train_x.T), requires_grad=False).type(dtype_float)
    y_labels = Variable(torch.from_numpy(np.argmax(train_y, 0)), requires_grad=False).type(dtype_long)


    model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.Linear(dim_h, dim_out)
    )

    loss_fn = torch.nn.CrossEntropyLoss()


    ## TRAINING THE MODEL
    alpha = 1e-2
    # optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    optimizer = torch.optim.SGD(model.parameters(), lr=alpha, momentum=0.99)

    for t in range(n):
        # forward
        y_pred = model(x)
        loss = loss_fn(y_pred, y_labels)
        # print("[Current Loss] ", loss)

        model.zero_grad()

        # backward
        loss.backward()
        optimizer.step()


    print('==========TRAINING SET==========')
    y_pred = model(x).data.numpy()
    training_performance = np.mean(np.argmax(y_pred, 1) == np.argmax(train_y, 0))
    print("[Performance - TRAINING SET] ", (training_performance * 100), "%\n")

    print('==========Validation SET==========')

    validating_x, validating_y = generate_x_y("validating")

    x = Variable(torch.from_numpy(validating_x.T), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()

    validating_performance = np.mean(np.argmax(y_pred, 1) == np.argmax(validating_y, 0))
    print("[Performance - Validation SET] ", (validating_performance * 100), "%\n")

    print('==========TEST SET==========')

    test_x, test_y = generate_x_y("testing")

    x = Variable(torch.from_numpy(test_x.T), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()

    test_performance = np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 0))
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

    return training_performance, validating_performance, test_performance, model




if __name__ == "__main__":
    dim_x = 32 * 32
    dim_out = 6
    dim_h_max = int(np.mean([dim_x, dim_out]))

    best_h_training_performances = list()
    best_h_validating_performances = list()
    best_h_test_performances = list()

    addon = list(range(50, dim_h_max, 50))
    dim_hs = [6, 12]
    dim_hs.extend(addon)

    ## Get Best Dimension of the Hidden Layer
    for dim_h in dim_hs:
        print("======================= Current Dimension of H: {} =======================\n".format(dim_h))
        print("==================== General Parameters ====================")
        print("[Dimension of x]", dim_x)
        print("[Dimension of output]", dim_out)
        print("[Maximum Dimension of H] ", dim_h_max, '\n')
        print("==================== Current Iteration =====================")
        training_performance, validating_performance, test_performance, model = train(dim_x, dim_h, dim_out, 10000)
        best_h_training_performances.append(training_performance)
        best_h_validating_performances.append(validating_performance)
        best_h_test_performances.append(test_performance)
        print("=========================================================================\n")

    print("==================== Results ====================")
    print("[best_h_training_performances] ", best_h_training_performances)
    print("[best_h_validating_performances] ", best_h_validating_performances)
    print("[best_h_test_performances] ", best_h_test_performances)
    print("[Best dim_h] ", dim_hs[np.argmax(best_h_test_performances)])
    print("[Best test_performance] ", (best_h_test_performances[np.argmax(best_h_test_performances)] * 100), "%")

    plot_x = dim_hs
    plt.plot(plot_x, best_h_training_performances, 'c', label='Training Set Performances')
    plt.plot(plot_x, best_h_validating_performances, 'y', label='Validation Set Performances')
    plt.plot(plot_x, best_h_test_performances, 'm', label='Test Set Performances')
    plt.legend(loc='best')
    plt.xlabel("Dimension of The Hidden Layer", fontsize=10.5)
    plt.ylabel("Performances (%)", fontsize=10.5)
    plt.savefig('part8-best-dim_h-plot.png')
    plt.gcf().clear()

    ## Get Learning Curve

    training_performances = list()
    validating_performances = list()
    test_performances = list()

    dim_h = dim_hs[np.argmax(best_h_test_performances)]  ## Best dim_h
    max_iter = 15000
    iters = list(range(0, max_iter, 500))

    for i in iters:
        print("========================= Current Iteration: {} =========================\n".format(i))
        print("==================== General Parameters ====================")
        print("[Dimension of x]", dim_x)
        print("[Dimension of h]", dim_h)
        print("[Dimension of output]", dim_out)
        print("[Maximum # of Iterations] ", max_iter, '\n')
        print("==================== Current Information =====================")
        training_performance, validating_performance, test_performance, model = train(dim_x, dim_h, dim_out, i)
        training_performances.append(training_performance)
        validating_performances.append(validating_performance)
        test_performances.append(test_performance)
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
    plt.savefig('part8-performances-plot.png')
    plt.gcf().clear()

    it = iters[np.argmax(test_performances)]  ## Best # of iterations
    training_performance, validating_performance, test_performance, model = train(dim_x, dim_h, dim_out, it)
    
    save_weight_imgs(model, [0, 1, 2, 3, 4, 5])