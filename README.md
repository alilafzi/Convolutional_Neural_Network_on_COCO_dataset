# Convolutional_Neural_Network_on_COCO_dataset

## COCO images downloader
The python script "hw04_coco_downloader.py" downloads images from the competition Microsoft COCO dataset (based on each of the available giant annotation files) and resizes them to 64x64 pixels. This script has a parser argument that gets the path to the main directory of COCO images, class list, the number of images in each class, and the path to the folder containing the desired annotation JSON file as its inputs while calling the script in the terminal command. <br>

## Training
"hw04_training.py" runs by providing the path to the main directory containing the COCO images and class list as its parser arguments. It also contains the necessary dataloader for feeding the downloaded images in the previous step and 3 types of networks to train them as described below: <br>
Net 1: a single convolutional layer followed by ReLU activation and max pooling and a single fully connected hidden layer before the output layer. <br>
Net 2: two convolutional layers, each followed by ReLU activation and max pooling and a single fully connected hidden layer before the output layer. <br>
Net 3: similar to Net 2 with the difference of having padding of 1 in the first conv. layer. <br>

The following image shows the comparison between the training loss of the aforementioned networks at every 500 iterations: <br>
<img src="https://github.com/alilafzi/Convolutional_Neural_Network_on_COCO_dataset/blob/main/images/train_loss.jpg" height = 400 width = 400> <br>

This script also saves all the learnable parameters of the trained networks as separate .pth files. <br>

## Testing
"hw04_validation.py" works similar to the training sript in terms of the parser arguments. It loads each of the saved trained networks from the previous task and plots the confusion matrix on the test set for each of them. <br><br>
<img src="https://github.com/alilafzi/Convolutional_Neural_Network_on_COCO_dataset/blob/main/images/net1_confusion_matrix.jpg" height = 400 width = 400> <br>
Net 1 confusion matrix <br><br>
<img src="https://github.com/alilafzi/Convolutional_Neural_Network_on_COCO_dataset/blob/main/images/net2_confusion_matrix.jpg" height = 400 width = 400> <br>
Net 2 confusion matrix <br><br>
<img src="https://github.com/alilafzi/Convolutional_Neural_Network_on_COCO_dataset/blob/main/images/net3_confusion_matrix.jpg" height = 400 width = 400> <br>
Net 3 confusion matrix <br>

## Dataset:
Class list: ["airplane", "boat", "cat", "dog", "elephant", "giraffe", "horse", "refrigerator", "train", "truck"] <br>
Training set: 3500 images for each class <br>
Test set: 500 images for each class <br>

## Reference:
https://github.com/cocodataset/cocoapi
