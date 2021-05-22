'''
Alex Costanzino, Marco Costante
MSc student in Artificial Intelligence
@ Alma Mater Studiorum, University of Bologna
March, 2021
'''

from pycocotools.coco import COCO
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from shutil import copyfile

# Choose the right path
annFile = './train/annotations.json'

# Initialize COCO api for instance annotations
coco = COCO(annFile)

# Display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(cats[0])))

image_directory = './train/images/'
mask_directory = './train/masks/'

images = os.listdir(image_directory)

coco_imgs = []

for i, image_name in tqdm(enumerate(images), total = len(images), position = 0, leave = True):
    imgId = int(coco.getImgIds(imgIds = [image_name.split('.')[0]])[0].lstrip("0"))
    coco_imgs.append(coco.loadImgs([imgId])[0])

'''Image categorization'''
for i, coco_img in tqdm(enumerate(coco_imgs), total = len(coco_imgs), position = 0, leave= True):
    annIds = coco.getAnnIds(imgIds=coco_img['id'], iscrowd = None)
    for ann in coco.loadAnns(annIds):
        catName = [cat['name'] for cat in cats if cat['id'] == ann['category_id']][0]
        if os.path.isfile(os.path.join(image_directory, coco_img['file_name'])):
            if not os.path.exists('./train_cat/images/' + catName + '/'):
                os.makedirs('./train_cat/images/' + catName + '/')
            copyfile(image_directory + coco_img['file_name'], './train_cat/images/' + catName + '/' + coco_img['file_name'])

fold_cats = os.listdir('./train_cat/images/')
for i, folder in tqdm(enumerate(fold_cats), total = len(fold_cats), position = 0, leave = True):
    if not os.path.isfile(folder):
        images = os.listdir('./train_cat/images/' + folder)
        coco_imgs = []
        for image_name in images:
            imgId = int(coco.getImgIds(imgIds = [image_name.split('.')[0]])[0].lstrip("0"))
            coco_imgs.append(coco.loadImgs([imgId])[0])
        for coco_img in coco_imgs:
            catId = coco.getCatIds(catNms=[folder])[0]
            annIds = coco.getAnnIds(imgIds=coco_img['id'], catIds = [catId], iscrowd = None)
            anns = coco.loadAnns(annIds)
            mask = np.zeros((coco_img['height'], coco_img['width']))
            for ann in anns:
                mask += coco.annToMask(ann)
            file_mask = Image.fromarray(mask*255)
            file_mask = file_mask.convert('RGB')
            catName = [cat['name'] for cat in cats if cat['id'] == ann['category_id']][0]
            if not os.path.exists('./train_cat/masks/' + catName + '/'):
                os.makedirs('./train_cat/masks/' + catName + '/')
            file_mask.save('./train_cat/masks/' + catName + '/' + coco_img['file_name'])