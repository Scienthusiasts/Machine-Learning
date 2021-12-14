import numpy as np
import matplotlib.pyplot as plt 


train_loss = np.load("./eval_param/train_loss.npy")
test_loss = np.load("./eval_param/test_loss.npy")
train_acc = np.load("./eval_param/train_acc.npy")
test_acc = np.load("./eval_param/test_acc.npy")

plt.subplot(211)
plt.plot(train_loss[100:], label = 'train loss')
plt.plot(test_loss[100:], label = 'test loss')
plt.xlabel('epoch')
plt.ylabel('CEloss value')
plt.legend()

plt.subplot(212)
plt.plot(train_acc, label = 'train acc')
plt.plot(test_acc, label = 'test acc')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

# train_loss_354 = np.load("train_loss_354.npy")
# test_loss_354 = np.load("test_loss_354.npy")

# train_loss_128 = np.load("train_loss_128.npy")
# test_loss_128 = np.load("test_loss_128.npy")

# train_loss_32 = np.load("train_loss_32.npy")
# test_loss_32 = np.load("test_loss_32.npy")

# train_loss_1 = np.load("train_loss_1.npy")
# test_loss_1 = np.load("test_loss_1.npy")


# ax1 = plt.subplot(221)
# ax1.plot(train_loss_354[5000:], label = 'train loss')
# ax1.plot(test_loss_354[5000:], label = 'test loss')
# ax1.set_title('batch size = total')
# ax1.set_xlabel('epoch')
# ax1.set_ylabel('MSEloss value')

# ax2 = plt.subplot(222)
# ax2.plot(train_loss_128[5000:])
# ax2.plot(test_loss_128[5000:])
# ax2.set_title('batch size = 128')
# ax2.set_xlabel('epoch')
# ax2.set_ylabel('MSEloss value')

# ax3 = plt.subplot(223)
# ax3.plot(train_loss_32[5000:])
# ax3.plot(test_loss_32[5000:])
# ax3.set_title('batch size = 32')
# ax3.set_xlabel('epoch')
# ax3.set_ylabel('MSEloss value')

# ax4 = plt.subplot(224)
# ax4.plot(train_loss_1[5000:])
# ax4.plot(test_loss_1[5000:])
# ax4.set_title('batch size = 1')
# ax4.set_xlabel('epoch')
# ax4.set_ylabel('MSEloss value')



plt.legend()
plt.show()