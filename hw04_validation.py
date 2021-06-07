import argparse
parser = argparse . ArgumentParser ( description = 'HW04 Training/Validation'
)
parser . add_argument ( '--root_path' , type = str , required
= True )
parser . add_argument ( '--class_list' , nargs = '*' , type = str ,
required = True )
args , args_other = parser . parse_known_args ()
#print (args.class_list)

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
val_dataset = your_dataset_class (args.root_path, args.class_list, transform, 'Val')
#print ('val size', val_dataset.__len__())
val_data_loader = torch.utils.data.DataLoader(dataset =
val_dataset ,
batch_size = val_dataset.__len__() ,
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

#Net1 architecture
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

##Net2 architecture
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

#Net3 architecture
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


os.chdir(args.root_path)
os.chdir('..')
os.chdir('Train')
#PATH = 'net1.pth'
#net1 = torch.load(PATH)
#net1.eval()
PATH = 'net1.pth'
net1 = TemplateNet1()
net1.load_state_dict(torch.load(PATH))
net1.eval()

PATH = 'net2.pth'
net2 = TemplateNet2()
net2.load_state_dict(torch.load(PATH))
net2.eval()

PATH = 'net3.pth'
net3 = TemplateNet3()
net3.load_state_dict(torch.load(PATH))
net3.eval()

#Net1 validation
predictions = []
for i, data in enumerate(val_data_loader):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = net1(inputs)
			
for i in range(outputs.shape[0]):
	predictions.append(torch.argmax(outputs[i]))

labels = np.array(labels)
predictions = np.array(predictions)

import seaborn as sns			
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
os.chdir(args.root_path)

score = accuracy_score(labels, predictions)

mat = confusion_matrix(labels, predictions)
index = args.class_list
columns = args.class_list
import pandas as pd
plt.figure(0)
mat_df = pd.DataFrame(mat, columns, index)
sns.heatmap(mat_df.T, square=True, annot=True, fmt = 'd', cbar = False)
plt.ylabel('Predicted label')
plt.xlabel('True label\nAccuracy={:0.4f}'.format(score))
plt.savefig('net1_confusion_matrix.jpg', bbox_inches = 'tight', pad_inches = 0)


#Net2 validation
predictions = []
for i, data in enumerate(val_data_loader):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = net2(inputs)
			
for i in range(outputs.shape[0]):
	predictions.append(torch.argmax(outputs[i]))

labels = np.array(labels)
predictions = np.array(predictions)

import seaborn as sns			
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
os.chdir(args.root_path)

score = accuracy_score(labels, predictions)

mat = confusion_matrix(labels, predictions)
index = args.class_list
columns = args.class_list
import pandas as pd
plt.figure(1)
mat_df = pd.DataFrame(mat, columns, index)
sns.heatmap(mat_df.T, square=True, annot=True, fmt = 'd', cbar = False)
plt.ylabel('Predicted label')
plt.xlabel('True label\nAccuracy={:0.4f}'.format(score))
plt.savefig('net2_confusion_matrix.jpg', bbox_inches = 'tight', pad_inches = 0)


#Net3 validation
predictions = []
for i, data in enumerate(val_data_loader):
			inputs, labels = data
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = net3(inputs)
			
for i in range(outputs.shape[0]):
	predictions.append(torch.argmax(outputs[i]))

labels = np.array(labels)
predictions = np.array(predictions)

import seaborn as sns			
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
os.chdir(args.root_path)

score = accuracy_score(labels, predictions)

mat = confusion_matrix(labels, predictions)
index = args.class_list
columns = args.class_list
import pandas as pd
plt.figure(2)
mat_df = pd.DataFrame(mat, columns, index)
sns.heatmap(mat_df.T, square=True, annot=True, fmt = 'd', cbar = False)
plt.ylabel('Predicted label')
plt.xlabel('True label\nAccuracy={:0.4f}'.format(score))
plt.savefig('net3_confusion_matrix.jpg', bbox_inches = 'tight', pad_inches = 0)




