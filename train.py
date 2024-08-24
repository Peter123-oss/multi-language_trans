import torch
import utils.loader as loader
import model.lenet4 as lenet4

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_images = loader.load_image_fromfile('data/train-images.idx3-ubyte')
test_images = loader.load_image_fromfile('data/t10k-images.idx3-ubyte')
train_labels = loader.load_label_fromfile('data/train-labels.idx1-ubyte')
test_labels = loader.load_label_fromfile('data/t10k-labels.idx1-ubyte')
train_images=torch.Tensor(train_images).view(train_images.shape[0],1,train_images.shape[1],train_images.shape[2])
test_images=torch.Tensor(test_images).view(test_images.shape[0],1,test_images.shape[1],test_images.shape[2])
train_labels=torch.LongTensor(train_labels)
test_labels=torch.LongTensor(test_labels)

train_dataset=torch.utils.data.TensorDataset(train_images,train_labels)
test_dataset=torch.utils.data.TensorDataset(test_images,test_labels)
model=lenet4.lenet()
cross=torch.nn.CrossEntropyLoss()
opt=torch.optim.Adam(model.parameters(),lr=0.001)

epoch=50
train_loader=torch.utils.data.DataLoader(dataset=train_dataset,shuffle=True,batch_size=128)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,shuffle=True,batch_size=128)


for e in range(epoch):
    correct_guess=0
    total=0
    for x,y in train_loader:
        opt.zero_grad()
        y_hat=model(x)
        loss=cross(y_hat,y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        for x,y in test_loader:
            y_hat=model(x)
            pre=torch.nn.functional.log_softmax(y_hat,1)
            pre=torch.argmax(pre,dim=1)
            correct_guess+=(pre==y).float().sum()
            total+=pre.shape[0]
    accuracy=correct_guess/total
    print(F"epoch: {e},  accuracy: {accuracy}")

    state_dict=model.state_dict()
    torch.save(state_dict,"lenet3.pth")
