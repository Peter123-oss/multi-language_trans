import torch
class lenet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer1=torch.nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(5,5),padding=2)
        self.layer2=torch.nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),padding=0)
        self.layer3=torch.nn.Conv2d(in_channels=16,out_channels=120,kernel_size=(5,5),padding=0)
        self.layer4=torch.nn.Linear(120,84)
        self.layer5=torch.nn.Linear(84,10)

    def forward(self,input):
        o1= self.layer1(input)
        o1=torch.nn.functional.relu(o1)
        o1=torch.nn.functional.max_pool2d(o1,kernel_size=(2,2))

        o2=self.layer2(o1)
        o2 = torch.nn.functional.relu(o2)
        o2 = torch.nn.functional.max_pool2d(o2, kernel_size=(2, 2))

        o3 = self.layer3(o2)
        o3 = torch.nn.functional.relu(o3)

        o3=o3.squeeze()
        o4=self.layer4(o3)
        o4 = torch.nn.functional.relu(o4)

        o5=self.layer5(o4)
        return o5




