#import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
#print(digits.DESCR)

#print(digits.data)
#print(digits.target)

# Figure size (width, height)
 
fig = plt.figure(figsize=(6, 6))
 
# Adjust the subplots 
 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
 
# For each of the 64 images
 
for i in range(64):
 
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
 
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
 
    # Display an image at the i-th position
 
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
 
    # Label the image with the target value
 
    ax.text(0, 7, str(digits.target[i]))
 
plt.show()

model = KMeans(n_clusters= 10, random_state=42)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
 
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
 
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()
new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,3.29,6.79,6.86,6.86,6.10,0.23,0.00,1.37,7.55,5.49,3.81,5.34,7.62,0.76,0.00,0.91,5.03,0.61,0.08,5.49,7.40,0.15,0.00,0.00,2.06,6.25,7.01,7.62,4.88,0.00,0.00,0.00,4.57,7.62,7.62,7.62,7.62,2.37,0.00,0.00,0.00,0.00,0.00,0.53,2.06,0.23,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,2.21,3.81,1.91,0.00,0.00,0.00,0.00,3.28,7.62,7.55,7.40,0.46,0.00,0.00,0.00,1.98,3.43,3.28,7.62,0.76,0.00,0.00,0.00,0.00,3.43,7.32,7.63,0.54,0.00,0.00,0.00,0.00,4.73,6.25,7.62,2.98,0.00,0.00,0.38,0.15,0.00,0.15,7.17,4.57,0.00,0.00,6.33,5.95,4.57,4.73,7.47,4.04,0.00,0.00,3.36,6.10,6.10,6.10,5.19,0.46,0.00,0.00],
[1.14,3.81,3.58,3.05,3.05,2.29,0.00,0.00,3.81,7.63,7.55,7.62,7.62,7.32,0.00,0.00,3.35,7.09,0.00,0.92,7.47,4.42,0.00,0.00,0.08,0.46,0.00,3.97,7.55,0.92,0.00,0.00,0.00,0.00,0.00,6.26,5.64,0.00,0.00,0.00,0.00,0.00,0.53,7.62,3.66,0.00,0.00,0.00,0.00,0.00,1.44,7.62,2.52,0.00,0.00,0.00,0.00,0.00,1.60,7.32,0.92,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,4.73,3.96,4.50,3.43,0.00,0.00,0.00,0.00,7.09,4.74,5.95,6.25,0.00,0.00,0.00,1.14,7.62,6.02,6.40,7.32,4.58,1.91,0.00,0.53,5.80,6.10,6.71,7.62,6.25,2.82,0.00,0.00,0.00,0.00,2.90,7.62,1.60,0.00,0.00,0.00,0.00,0.00,1.53,7.62,2.29,0.00,0.00,0.00,0.00,0.00,1.53,7.62,2.52,0.00]
])

new_labels = model.predict(new_samples)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')