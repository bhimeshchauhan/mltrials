# import numpy as np
# import matplotlib.image as mpimg
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# img = mpimg.imread('plant_color.png')
# pixels = img.shape[0]*img.shape[1]
# channels = 3
# data = np.reshape(img[:, :, :channels], (pixels, channels))

# histo_rgb, _ = np.histogramdd(data, bins=256)
# r, g, b = np.nonzero(histo_rgb)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter3D(r, g, b)
# ax.set_xlabel('Red')
# ax.set_ylabel('Green')
# ax.set_zlabel('Blue')
# plt.title('RGB colors')
# plt.show()



import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img = mpimg.imread('test3.jpg')
pixels = img.shape[0]*img.shape[1]
channels = 3
data = np.reshape(img[:, :, :channels], (pixels, channels))

histo_rgb, _ = np.histogramdd(data, bins=256)
histo_rg = np.sum(histo_rgb, 2)
levels = np.arange(256)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for g in levels:
    ax.bar(levels, histo_rg[:, g], zs=g, zdir='y', color='r')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Number of pixels')
plt.show()