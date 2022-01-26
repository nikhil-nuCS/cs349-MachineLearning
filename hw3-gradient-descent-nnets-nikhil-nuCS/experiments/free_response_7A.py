#from sklearn.neural_network import MLPClassifier
#from sklearn.datasets import fetch_openml
#from sklearn.model_selection import train_test_split
#import numpy as np
#
#
#features, targets = fetch_openml("mnist_784", version=1, return_X_y=True)
#features /= 255.0
#
#train_features, test_features = features[:60000], features[60000:]
#train_targets, test_targets = targets[:60000], targets[60000:]
#
#means = []
#std = []
#for i in [1,4,16,64,256]:
#    accuracies = []
#    for j in range(10):
#        classifier = MLPClassifier(hidden_layer_sizes=(i,))
#        classifier.fit(train_features,train_targets)
#        accuracies.append(classifier.score(test_features,test_targets))
#    means.append(np.mean(accuracies))
#    std.append(np.std(accuracies))
#print('Mean')
#print(means)
#print('Standard deviation')
#print(std)
