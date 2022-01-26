from your_code import GradientDescent, load_data
from your_code import L1Regularization, L2Regularization
from your_code import HingeLoss, SquaredLoss, ZeroOneLoss
from your_code import accuracy, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    #QUESTION 3
    #PART-A
    #CT
    file_name = 'mnist-binary'
    max_iterions = 2000
    batch_size = None
    fraction = 1.0
    learning_rate = 1e-5
    reg_param = 0.05
    loss = 'squared'
    regularization = None
    epsilon = 1e-3

    train_features, test_features, train_targets, test_targets = load_data(file_name, fraction=fraction)
    lambda_list = [1e-3, 1e-2, 1e-1, 1, 10, 100]
    list_of_losses_l1 = []
    list_of_losses_l2 = []
    
    for lamba in lambda_list:
        learner_l1 = GradientDescent(loss=loss, regularization='l1', learning_rate=learning_rate, reg_param=lamba)
        learner_l1.fit(train_features, train_targets, batch_size, max_iterions)
        
        print("L1 Finished L2 Start")

#        learner_l2 = GradientDescent(loss=loss, regularization='l2', learning_rate=learning_rate, reg_param=lamba)
#        learner_l2.fit(train_features, train_targets, batch_size, max_iterions)

#        for index in range(len(learner_l1.model)):
#            if(abs(learner_l1.model[index]) > epsilon):
#                non1+=1
#
#            if(abs(learner_l2.model[index]) > epsilon):
#                non2+=1
#
#        list_of_losses_l1.append(non1)
#        list_of_losses_l2.append(non2)


        learner_l2 = GradientDescent(loss=loss, regularization='l2', learning_rate=learning_rate, reg_param=lamba)
        learner_l2.fit(train_features, train_targets, batch_size, max_iterions)
        list_of_losses_l1.append(np.sum(np.where(abs(learner_l1.model) > epsilon, 1, 0)))
        list_of_losses_l2.append(np.sum(np.where(abs(learner_l2.model) > epsilon, 1, 0)))
        
    plt.figure()
    plt.plot(range(len(lambda_list)), list_of_losses_l1, label='L1')
    plt.plot(range(len(lambda_list)), list_of_losses_l2, label='L2')
    plt.legend()
    plt.title('Number of Nonzero Weights vs. Lambda')
    plt.xlabel('Lambda')
#    plt.xscale('log')
    plt.xticks([0,1,2,3,4,5], ['1e-3', '1e-2', '1e-1', '1', '10', '100'])
    plt.ylabel('Number of Nonzero W')
    plt.savefig('experiments/3_a_ct_graph_rks.png')
    
#    plt.figure()
#    plt.plot(lambda_list, list_of_losses_l1, color='orange', label='L1')
#    plt.plot(lambda_list, list_of_losses_l2, color='blue', label='L2')
#    plt.title('Number of Non-Zero Model Weights Vs. Lambda (log-scale)')
#    plt.xlabel('Lambda (log-scale)')
#    plt.xscale('log')
#    plt.ylabel('Number of Non-Zero Model Weights')
#    plt.legend(loc="best")
#    plt.savefig("experiments/4_a_yb.png")

     
     
     
     
    
    
    #YB
#    train_features, test_features, train_targets, test_targets = load_data(file_name, fraction=fraction)
#    lambda_list = [1e-3, 1e-2, 1e-1, 1, 10, 100]
#    list_of_losses_l1 = []
#    list_of_losses_l2 = []
#
#    for lamba in lambda_list:
#
#        learner_l1 = GradientDescent(loss=loss, regularization='l1', learning_rate=learning_rate, reg_param=lamba)
#        learner_l1.fit(train_features, train_targets, batch_size, max_iterions)
#        list_of_losses_l1.append()
#        list_of_losses_l1.append(np.sum(np.where(abs(learner_l1.model) > epsilon, 1, 0)))
#
#        learner_l2 = GradientDescent(loss=loss, regularization='l2', learning_rate=learning_rate, reg_param=lamba)
#        learner_l2.fit(train_features, train_targets, batch_size, max_iterions)
#        list_of_losses_l2.append(np.sum(np.where(abs(learner_l2.model) > epsilon, 1, 0)))
#
#     plt.figure()
#     plt.plot(range(len(lambda_list)), list_of_losses_l1, label='l1')
#     plt.plot(range(len(lambda_list)), list_of_losses_l2, label='l2')
#     plt.legend()
#     plt.title('Number of Nonzero Weights vs. Lambda')
#     plt.xlabel('lambda')
#     plt.xticks([0,1,2,3,4,5], lambda_list)
#     plt.ylabel('Number of Nonzero W')
#     plt.savefig('experiments/4_a_ct.png')
#
