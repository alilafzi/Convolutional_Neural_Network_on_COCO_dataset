''' Part of this code is borrowed from the given reference in the homework handout, i.e. github.com/cocodataset/cocoapi  '''
import glob
import os
import numpy
import PIL
import argparse 
import requests
import logging
import json

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

parser = argparse.ArgumentParser ( description = 'HW04 COCO downloader')
parser.add_argument ( '--root_path'  , type = str, required = True )
parser.add_argument ( '--images_per_class' , type = int ,required = True) 
parser.add_argument ( '--coco_json_path' , type = str , required =True )
parser.add_argument ( '--class_list' , nargs = '*', type = str , required =True )
args , args_other = parser.parse_known_args ()

from requests.exceptions import ConnectionError, ReadTimeout, TooManyRedirects, MissingSchema, InvalidURL
from PIL import Image
from urllib.request import urlretrieve

classes_to_scrape = args.class_list

if not os.path.exists(args.root_path):
		os.mkdir(args.root_path)

dataDir = args.coco_json_path
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)
	
for i in range(len(classes_to_scrape)):
	catIds = coco.getCatIds(catNms=[classes_to_scrape[i]]);
	imgIds = coco.getImgIds(catIds=catIds );
	imgs = coco.loadImgs(imgIds)

	counter = 0
	class_name = classes_to_scrape[i]
	
	class_folder = os.path.join(args.root_path, class_name)
	if not os.path.exists(class_folder):
		os.mkdir(class_folder)

	os.chdir(args.root_path)	
	for im in imgs:
		counter += 1
		if counter > args.images_per_class:
			break
		
		os.chdir(class_folder)			
		filenames_before_download = os.listdir(class_folder)
		
		img_data = requests.get(im['coco_url']).content
		with open(im['file_name'], 'wb') as downloader:			
			downloader.write(img_data)

		img_file_path = os.path.join(class_folder, im['file_name'])
		# Resize image to 64x64
		im = Image . open ( img_file_path )

		if im.mode != "RGB":
			im = im.convert ( mode = "RGB" )

		im_resized = im . resize (( 64 , 64 ) , Image . BOX )
		# Overwrite original image with downsampled image
		im_resized . save ( img_file_path )	
		##check if an image is already downloaded
		filenames_after_download = os.listdir(class_folder)
		
		if len(filenames_before_download) == len(filenames_after_download):
			counter -= 1
