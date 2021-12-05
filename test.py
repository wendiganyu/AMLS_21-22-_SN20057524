import numpy as np
from sklearn.decomposition import PCA
import pickle as pk
import matplotlib.pyplot as plt


import PreProcessing

# pca_dims = PCA()
# pca_dims.fit(x_train)

x_train, x_valid, y_train, y_valid = PreProcessing.gen_train_test_set(is_mul=False, random_state=109)
print(x_train[1])
pca = pk.load(open("tmp/pca.pkl", 'rb'))
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print(d)
X_reduced = pca.transform(x_train)
X_valid_reduced = pca.transform(x_valid)
X_recovered = pca.inverse_transform(X_reduced)
X_valid_recovered = pca.inverse_transform(X_valid_reduced)
print("reduced shape: " + str(X_reduced.shape))
print("recovered shape: " + str(X_recovered.shape))

print("reduced valid shape: " + str(X_valid_reduced.shape))
print("recovered valid shape: " + str(X_valid_recovered.shape))

f = plt.figure()
f.add_subplot(1,2, 1)
plt.title("original")
plt.imshow(x_train[0].reshape((512,512)), cmap="gray")
f.add_subplot(1,2, 2)

plt.title("PCA compressed")
plt.imshow(X_recovered[0].reshape((512,512)), cmap="gray")
plt.show(block=True)


f = plt.figure()
f.add_subplot(1,2, 1)
plt.title("original")
plt.imshow(x_valid[0].reshape((512,512)), cmap="gray")
f.add_subplot(1,2, 2)

plt.title("PCA compressed")
plt.imshow(X_valid_recovered[0].reshape((512,512)), cmap="gray")
plt.show(block=True)


# plt.imshow(np.reshape(x_train[1], (512, 512)), cmap="gray")
# plt.show()
# pk.dump(pca_dims, open("tmp/pca.pkl", "wb"), protocol=4)

