from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
import numpy as np


# We load the data with load_iris from sklearn
data = load_iris()
features = data['data']
features_bk = data['data']
feature_names = data['feature_names']
target = data['target']
target_name = data['target_names']
labels = np.array([target_name[i] for i in target])

for t,marker,c in zip(range(3),">ox","rgb"):
    #We plot each class on its own to get different colored markers
    plt.scatter(features[target == t, 0],
                features[target == t, 2],
                marker=marker,
                c=c)
plt.show()


plength = features[:,2]
#use numpy operation to get setosa features
is_setosa = (target == 0)
#This is the important step:
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print('Maximum of setosa: {0}.'.format(max_setosa))
print('Minimum of others: {0}.'.format(min_non_setosa))

features = features[~is_setosa]
labels = labels[~is_setosa]
virginica = (labels == 'virginica')

best_acc = -1.0
best_fi = 0.0
best_t = 0.0
for fi in range(features.shape[1]):
    #We are going to generate all possible threshold for this feature
    thresh = features[:,fi].copy()
    thresh.sort()
    # Now test all thresholds:
    for t in thresh:
        pred = (features[:,fi] > t)
        acc = (pred == virginica).mean()
        if acc > best_acc:
            best_acc = acc
            best_fi = fi
            best_t = t

print(best_acc)
print(best_fi)
print(best_t)

for t,marker,c in zip(range(1,3),"ox","gb"):
    #We plot each class on its own to get different colored markers
    plt.scatter(features_bk[target == t,3],
                features_bk[target == t,2],
                marker=marker,
                c=c)
plt.xlabel(feature_names[3])
plt.ylabel(feature_names[2])
plt.show()

error = 0.0
for ei in range(len(features)):
    #select all but the one at position 'ei'
    training = np.ones(len(features),bool)
    training[ei] = False
    testing = ~training
    model = learn_model(features[training],virginica[training])
    predictions = apply_model(features[testing], virginica[testing], model)
    error += np.sum(predictions != virginica[testing])