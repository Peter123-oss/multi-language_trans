import torch
import model.lenet4 as Lenet
import cv2
model=Lenet.lenet()
state=torch.load("./lenet3.pth")
model.load_state_dict(state_dict=state)

image=cv2.imread("./5.png")
image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image=cv2.resize(image,(28,28))
image=255-image
ima=torch.Tensor(image).view(1,1,image.shape[0],image.shape[1])
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # =====>1 * 1 * 28 * 28
# image = cv2.resize(image,(28,28))
# # 28*28  图片形状
# # 增加维度 4维度 转换为张量
# img = torch.Tensor(image).view(1,1,image.shape[0],image.shape[1])
# # 10
# print(img.shape)

y_hat=model(ima)
y_hat = torch.nn.functional.log_softmax(y_hat, 0)
print(y_hat)
predict = torch.argmax(y_hat, dim=0)  # 10000*1
print(predict)
print(predict.numpy())
print(ima.shape)

