import pickle
from matplotlib import pyplot as plt

from cryoio import mrc

# with open('./data/1AON_test.pkl', 'rb') as pkl_file:
#     test_imgs = pickle.load(pkl_file)

# print(test_imgs.shape)
# test_data = test_imgs.reshape([-1, 64, 64])
# plt.figure(1)
# for i in range(9):
#     plt.subplot(331+i)
#     plt.imshow(test_data[i])
# plt.show()

test = mrc.readMRCimgs('./data/EMD-6044/EMD6044_test.mrcs', 0)
print(test.shape)
plt.figure(1)
for i in range(9):
    plt.subplot(331+i)
    plt.imshow(test[:, :, i])
plt.show()