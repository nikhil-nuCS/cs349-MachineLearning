from your_code import GradientDescent, load_data
from your_code import L1Regularization, L2Regularization
from your_code import HingeLoss, SquaredLoss, ZeroOneLoss
from your_code import accuracy, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #QUESTION 2
    #PART-A
    #CT
    file_name = 'synthetic'
    max_iterions = 1000
    batch_size = None
    fraction = 1.0
    learning_rate = 1e-4
    reg_param = 0.05
    loss = ZeroOneLoss()
    regularization = None

#    train_features, test_features, train_targets, test_targets = load_data(file_name, fraction=fraction)
#    biases = np.linspace(-5.5, 0.5, 100)
#    initial_weight = np.ones((train_features.shape[1], 1))
#    train_features = np.append(train_features, np.ones((len(train_features), 1)), axis=1)
#
#    list_of_losses = []
#    curr_index = 0
#    for bias in biases:
#        weight = np.append(initial_weight, bias)
#        curr_loss = loss.forward(train_features, weight, train_targets)
#        list_of_losses.append(curr_loss)
#
#    plt.figure()
#    plt.plot(biases, list_of_losses)
#    plt.xlabel('Bias')
#    plt.ylabel('Loss')
#    plt.title('Loss vs. Bias')
##    plt.legend(loc="best")
#    plt.savefig('experiments/2_a_ct_graph.png')
    
    
    #YB
    train_features, test_features, train_targets, test_targets = load_data(file_name, fraction=fraction)
    biases = np.linspace(-5.5, 0.5, 100)
    np_ones = np.ones((train_features.shape[0], 1))
    train_features = np.append(train_features, np_ones, axis=1)
    initial_weight = np.array([1, 0])

    list_of_losses = []
    curr_index = 0
    for bias in biases:
        initial_weight[1] = bias
        curr_loss = loss.forward(train_features, initial_weight, train_targets)
        list_of_losses.append(curr_loss)

    plt.figure()
    plt.plot(biases, list_of_losses)
    plt.xlabel('Bias')
    plt.ylabel('Loss')
    plt.title('Loss vs. Bias')
#    plt.legend(loc="best")
    plt.savefig('experiments/2_a_yb_graph_nk.png')




    #QUESTION 2
    #PART-A
    #CT
    file_name = 'synthetic'
    max_iterions = 1000
    batch_size = None
    fraction = 1.0
    learning_rate = 1e-4
    reg_param = 0.05
    loss = ZeroOneLoss()
    regularization = None

#    train_features, test_features, train_targets, test_targets = load_data(file_name, fraction=fraction)
#    biases = np.linspace(-5.5, 0.5, 100)
#
#    train_features = train_features[[0,1,3,4]]
#    train_targets = train_targets[[0,1,3,4]]
#
#    initial_weight = np.ones((train_features.shape[1], 1))
#    train_features = np.append(train_features, np.ones((len(train_features), 1)), axis=1)
#
#    list_of_losses = []
#    for bias in biases:
#        weight = np.append(initial_weight, bias)
#        curr_loss = loss.forward(train_features, weight, train_targets)
#        list_of_losses.append(curr_loss)
#
#    plt.figure()
#    plt.plot(biases, list_of_losses)
#    plt.xlabel('Bias')
#    plt.ylabel('Loss')
#    plt.title('Landscape on a set of 4 points')
##    plt.legend(loc="best")
#    plt.savefig('experiments/2_b_ct_graph.png')


    #YB
    train_features, test_features, train_targets, test_targets = load_data(file_name, fraction=fraction)
    biases = np.linspace(-5.5, 0.5, 100)

    train_features = train_features[[0, 1, 4, 5]]
    train_targets = train_targets[[0, 1, 4, 5]]

    np_ones = np.ones((train_features.shape[0], 1))
    train_features = np.append(train_features, np_ones, axis=1)

    initial_weight = np.array([1, 0])

    list_of_losses = []
    curr_index = 0
    for bias in biases:
        initial_weight[1] = bias
        curr_loss = loss.forward(train_features, initial_weight, train_targets)
        list_of_losses.append(curr_loss)

    plt.figure()
    plt.plot(biases, list_of_losses)
    plt.xlabel('Bias')
    plt.ylabel('Loss')
    plt.title('Landscape on set of 4 points')
#    plt.legend(loc="best")
    plt.savefig('experiments/2_b_yb_graph_nk.png')

