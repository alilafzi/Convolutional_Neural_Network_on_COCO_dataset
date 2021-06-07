import argparse
parser = argparse . ArgumentParser ( description = 'HW04 Training/Validation'
)
parser . add_argument ( '--root_path' , type = str , required
= True )
parser . add_argument ( '--class_list' , nargs = '*' , type = str ,
required = True )
args , args_other = parser . parse_known_args ()
#print (args.root_path)

import torch
from torch . utils . data import DataLoader , Dataset
import scipy
from scipy import misc
import os
import glob
import PIL
from PIL import Image
import numpy as np
from skimage import io, transform
import torch.nn.functional as F

class your_dataset_class ( Dataset ) :
	def __init__ (self, root_path, class_list, transform, goal) :
		self.path = root_path
		self.classes = class_list
		self.transform = transform 
		self.goal = goal
		#main_directory = os.path.join(self.path, self.goal)
		self.x_data = []
		self.y_data = []
		#label = torch.zeros(len(self.classes))
		for i in range(0, len(self.classes)):
			os.chdir(self.path)
			os.chdir(self.classes[i])
			#label = torch.zeros(len(self.classes))
			#label[i] = 1
			label = i
			#folders = os.listdir(".")
			#for sub in folders:
			#os.chdir(sub)
			files = os.listdir(os.getcwd())
			for filename in files:	
				self.x_data.append(self.transform(Image.open(filename)))
				self.y_data.append(label)
			#os.chdir('..')
		self.len = len(self.x_data)

	def __getitem__ (self, index) :
		#print (self.x_data[index], self.y_data[index])
		return self.x_data[index], self.y_data[index] 

	def __len__ (self):
		#print (self.len)
		return self.len

#if __name__ == "__main__":
	#your_dataset_class ( args.imagenet_root, args.class_list ).__len__()

from torchvision import transforms as tvt
transform = tvt . Compose ( [ tvt . ToTensor () , tvt . Normalize (( 0.5 ,0.5 ,0.5 ) , ( 0.5 , 0.5 , 0.5 ) ) ] )
train_dataset = your_dataset_class (args.root_path, args.class_list, transform, 'Train')
#print ('train size', train_dataset.__len__())
train_data_loader = torch.utils.data.DataLoader(dataset =
train_dataset ,
batch_size =  10, #train_dataset.__len__(),
shuffle = True ,
num_workers = 4 )

import torch
import torch.nn as nn
import random
import matplotlib.pylab as plt

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.becnhmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)

dtype = torch . float64

device = torch . device ( "cuda:0" if torch.cuda.is_available() else "cpu" )

##Net1
class TemplateNet1(nn.Module):
	def __init__(self):
		super(TemplateNet1, self).__init__()
		self.conv1 = nn.Conv2d(3, 128, 3)
		self.conv2 = nn.Conv2d(128, 128, 3)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(128 * 31 * 31, 1000)
		self.fc2 = nn.Linear(1000, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		#x = self.pool(F.relu(self.conv2(x)))
## (D)
		x = x.view(-1, 128 * 31 * 31)
## (E)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

def run_code_for_training(net1):
	net1 = net1.to(device)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net1.parameters(), lr=1e-3, momentum=0.9)
	epochs = 10
	final_loss = []
	for epoch in range(epochs):
		running_loss = 0.0
		for i, data in enumerate(train_data_loader):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = net1(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			#final_loss.append(running_loss)
			if (i+1) % 500 == 0:
				print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(500)))
				final_loss.append(running_loss / float(500))
				running_loss = 0.0
	return final_loss, net1

net1 = TemplateNet1()
os.chdir(args.root_path)
import numpy as np
print('Net1')
loss, net1 = run_code_for_training(net1)

PATH = 'net1.pth'
torch.save(net1.state_dict(), PATH)

plt.figure(0)
plt.plot(np.arange(1, len(loss)+1), loss, label = 'Net1 Training Loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_loss.jpg', bbox_inches = 'tight', pad_inches = 0)

#PATH = 'net1.pth'
#torch.save(net1, PATH)

##Net2
class TemplateNet2(nn.Module):
	def __init__(self):
		super(TemplateNet2, self).__init__()
		self.conv1 = nn.Conv2d(3, 128, 3)
		self.conv2 = nn.Conv2d(128, 128, 3)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(128 * 14 * 14, 1000)
		self.fc2 = nn.Linear(1000, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
## (D)
		x = x.view(-1, 128 * 14 * 14)
## (E)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

def run_code_for_training(net2):
	net2 = net2.to(device)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net2.parameters(), lr=1e-3, momentum=0.9)
	epochs = 10
	final_loss = []
	for epoch in range(epochs):
		running_loss = 0.0
		for i, data in enumerate(train_data_loader):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = net2(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			#final_loss.append(running_loss)
			if (i+1) % 500 == 0:
				print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(500)))
				final_loss.append(running_loss / float(500))
				running_loss = 0.0
	return final_loss, net2

net2 = TemplateNet2()
os.chdir(args.root_path)
import numpy as np
print('Net2')
loss, net2 = run_code_for_training(net2)

PATH = 'net2.pth'
torch.save(net2.state_dict(), PATH)

plt.figure(0)
plt.plot(np.arange(1, len(loss)+1), loss, label = 'Net2 Training Loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_loss.jpg', bbox_inches = 'tight', pad_inches = 0)


##Net3
class TemplateNet3(nn.Module):
	def __init__(self):
		super(TemplateNet3, self).__init__()
		self.conv1 = nn.Conv2d(3, 128, 3, padding = 1)
		self.conv2 = nn.Conv2d(128, 128, 3)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(128 * 15 * 15, 1000)
		self.fc2 = nn.Linear(1000, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
## (D)
		x = x.view(-1, 128 * 15 * 15)
## (E)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x

def run_code_for_training(net3):
	net3 = net3.to(device)
	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(net3.parameters(), lr=1e-3, momentum=0.9)
	epochs = 10
	final_loss = []
	for epoch in range(epochs):
		running_loss = 0.0
		for i, data in enumerate(train_data_loader):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()
			outputs = net3(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			#final_loss.append(running_loss)
			if (i+1) % 500 == 0:
				print("\n[epoch:%d, batch:%5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / float(500)))
				final_loss.append(running_loss / float(500))
				running_loss = 0.0
	return final_loss, net3

net3 = TemplateNet3()
os.chdir(args.root_path)
import numpy as np
print('Net3')
loss, net3 = run_code_for_training(net3)

PATH = 'net3.pth'
torch.save(net3.state_dict(), PATH)

plt.figure(0)
plt.plot(np.arange(1, len(loss)+1), loss, label = 'Net3 Training Loss')
plt.xlabel('iterations')
plt.ylabel('loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('train_loss.jpg', bbox_inches = 'tight', pad_inches = 0)

