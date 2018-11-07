import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
#can't use import torch.autograd.Variable,must be model,can't import class
tran=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
traindataset=torchvision.datasets.CIFAR10(root='./',train=True,download=False,transform=tran)
trainloader=torch.utils.data.DataLoader(traindataset,batch_size=4,shuffle=True,num_workers=0)
testdataset=torchvision.datasets.CIFAR10(root='./',train=False,download=False,transform=tran)
testloader=torch.utils.data.DataLoader(testdataset,batch_size=4,shuffle=False,num_workers=0)


class Net(nn.Module):
	def __init__(self):
		super(Net,self).__init__()
		self.conv1=nn.Conv2d(3,6,5)
		self.pool=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(6,16,5)
		self.fc1=nn.Linear(16*5*5,120)
		self.fc2=nn.Linear(120,84)
		self.fc3=nn.Linear(84,10)
	def forward(self,x):            #must 4d batchsize*chanels*h*w
		x=self.pool(f.relu(self.conv1(x)))
		x=self.pool(f.relu(self.conv2(x)))
		x=x.view(-1,16*5*5)
		x=f.relu(self.fc1(x))
		x=f.relu(self.fc2(x))
		x=self.fc3(x)
		return x

def train(net):
	loss_function=nn.CrossEntropyLoss()
	optimer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
	for i in range(2):
		loss_num=0.0
		for j,datas in enumerate(trainloader,0):
			imgs,labels=datas
			inputs,input_labels=Variable(imgs),Variable(labels)

			out=net(inputs)
			loss=loss_function(out,input_labels)
			optimer.zero_grad()
			loss.backward() #loss is tensor,but it is not array,just a number,
                            #so loss.size()=torch.Size([])
                            #but loss.item() is successful
			optimer.step()
			loss_num+=loss.data.item()
			if j%2000==1999:
				print('(%d,%d) loss is %f'%(i,j,loss_num/2000))
				loss_num=0.0
	print('finished training')
def accurity_all(net):
	total=0.0
	num=0.0
	for datas in testloader:
		imgs,labels=datas
		inputs,input_labels=Variable(imgs),Variable(labels)
		out=net(inputs)
		_, out=torch.max(out,1)
		total+=(out==input_labels).sum().item()
		num+=input_labels.data.size(0)
	print('accurity on test_dataset is %f'%(total/num))
def accurity_class(net):
	correct_list=list(0. for i in range(10))
	total_list=list(0. for i in range(10))
	for datas in testloader:
		imgs,labels=datas
		out=net(Variable(imgs))
		_, out=torch.max(out,1)  #max函数输出元组，需要在前面加一个_,
		input_labels=Variable(labels)
		c=(out==input_labels).squeeze() #== need Tensor or python number
		for i in range(input_labels.data.size(0)):
			label_index=input_labels[i]
			correct_list[label_index]+=c[i].item() #Tensor’[] operation will get 
                                                    #a Tensor,you can also use [][].
                                                    #item() convert Tensor which
                                                    # only have a element
                                                    #to python number.
                                                    #python number add Tensor get
                                                    #Tensor,so we should use item() 
                                                    #to convert Tensor.
			total_list[label_index]+=1
	for i in range(10):
		print('class 1\'s accurity is %f'%(correct_list[i]/total_list[i]))
#save module and load module
def save_module_parameters(model):
    torch.save(model.state_dict(),'parameters.pkl')
def load_module_parameters():
    net=Net()
    net.load_state_dict(torch.load('parameters.pkl'))
    return net
def save_module(model):
    torch.save(model,'model.pkl')
def load_module():
    return torch.load('model.pkl')

#main
net=Net()
train(net)
print()
accurity_all(net)
print()
accurity_class(net)

#save and load model
save_module_parameters(net)
net_load=load_module_parameters()
print('load parameters:')
accurity_all(net_load)
print()
accurity_class(net_load)

#load whole model,it don't need Net source code,it can run
save_module(net)
model_load=load_module()
print('load model:')
accurity_all(model_load)
print()
accurity_class(model_load)