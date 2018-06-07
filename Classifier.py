import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

print sklearn.__version__
iris = load_iris()
'''
print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0]

print range(len(iris.target))
for i in range(len(iris.target)):
  print iris.data[i],iris.target[i]
  
'''

test_idx = [0,50,100]

# training  data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]


# K Neighbors Classifiers
model = KNeighborsClassifier(n_neighbors=1)
model.fit(train_data, train_target)


print test_target
print model.predict(test_data)

