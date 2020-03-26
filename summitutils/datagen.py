import numpy as np
import matplotlib.pylab as plt
import detectron2
import cv2
import os
import pathlib
import json
import random
from PIL import Image, ImageDraw, ImageDraw2
import pandas as pd
import torchvision
from torchvision import transforms
import torch
import shutil
import glob

def generate_background_image(color_center, color_edge, size=(512,512), generate=True, bkg_loc=None):
    if not generate:
        if bkg_loc is None:
            raise ValueError('Please pass bkg_loc if generate is False')
        img = retrieve_background_image(bkg_loc).resize(size)
        return img
    
    #distance calculation
    row_idx = np.repeat(np.arange(0, size[0]), size[1], axis=0).reshape(size)
    col_idx = np.repeat(np.arange(0, size[1]), size[0], axis=0).reshape((size[1], size[0])).T    
    
    center = (size[0]/2, size[1]/2)
    dist = np.zeros(size)
    dist = np.sqrt((row_idx - center[0])**2 + (col_idx - center[1])**2)
    max_dist = np.max(dist)
    dist /= max_dist    
    
    #colors
    r = color_edge[0] * dist + color_center[0] * (1-dist)
    g = color_edge[1] * dist + color_center[1] * (1-dist)
    b = color_edge[2] * dist + color_center[2] * (1-dist)    
    
    #check artifacts with Image.fromarray()
    img = Image.new('RGBA', size)
    for y in range(size[1]):
        for x in range(size[0]):
            img.putpixel((x,y), (int(r[x,y]), int(g[x,y]), int(b[x,y])))
    
    return img

def retrieve_background_image(LOC):
    bkg_files = glob.glob(f'{LOC}/*.jpg')
    
    img_filename = np.random.choice(bkg_files)
    
    img = Image.open(img_filename).convert('RGBA')

    return img

def make_logo_transparent(logo, threshold_color=[200,200,200]):
    logo = logo.convert('RGBA')
    data = logo.getdata()
    
    data_alpha = []
    for pixel in data:
        #if pixel[0]>200 and pixel[1]>200 and pixel[2]>200:
        if pixel[0]>threshold_color[0] and pixel[1] > threshold_color[1] and pixel[2] > threshold_color[2]:
            data_alpha.append((pixel[0], pixel[1], pixel[2], 0))
            #data_alpha.append((0,255,0,0))
        else:
            data_alpha.append((pixel[0], pixel[1], pixel[2], pixel[3]))
        
    logo.putdata(data_alpha)
    
    return logo

def generate_augmented_image(size=(512,512),
                             logo_file='CLOUDERALOGO.jpg',
                             N_tfms_to_apply=2,
                             low_frac=0.30,
                             high_frac=0.90,
                             grayscale=False,
                             generate=True,
                             bkg_loc=None,
                             perspective_tfm=False):
        
    logo = Image.open(logo_file) #inefficient to open N times but okay for now

    #logo
    #step 1: get resize parameters
    frac0 = (np.random.random()*(high_frac-low_frac) + low_frac)
    frac1 = (np.random.random()*(high_frac-low_frac) + low_frac)
    #logo_size = (int(frac0*size[0]), int(frac1*size[1]))
    
    aspect_ratio_distort = np.random.random()*0.2 + 0.9 #0.9*1.1
    logo_size = (int(aspect_ratio_distort*frac0*size[0]), int(aspect_ratio_distort*frac0*size[1])) #keep aspect ratio

    #logo_size = (128,128) #introduce some randomness

    #step 2: get mask and apply resizing
    logo = make_logo_transparent(logo)
    logo = logo.resize(logo_size)

    if perspective_tfm:
        logo = transforms.RandomPerspective(distortion_scale=0.7)(logo)
    
    #step 3: apply transformations
    tfms_list = [                
                    transforms.RandomRotation(45, expand=False, fill=255),
                    #transforms.RandomAffine(0, translate=(0.3,0.3)), #for truncations
                    transforms.RandomAffine(0, shear=10)
                ]

    N_tfms = len(tfms_list)

    logo_tfms = logo.copy()
    for i in range(N_tfms_to_apply): #apply N transformations
        logo_tfms = tfms_list[np.random.randint(N_tfms)](logo_tfms)
        #logo_tfms = make_logo_transparent(logo_tfms)

    logo_tfms = make_logo_transparent(logo_tfms)

    #step 4: background image 
    colors = [(255,0,0), (0,255,0), (0,0,255), (128,0,0), (0,128,0), (0,0,128), 
              (0,0,0), (255,255,255), (128,128,128),        
              (255,255,0), (255,0,255), (0,255,255), (128,128,0), (128,0,128), (0,128,128)]
    N_colors = len(colors)

    colors_picked = np.array(colors)[np.random.randint(0, N_colors, size=2)]

    bkg = generate_background_image(colors_picked[0], colors_picked[1], size=size, generate=generate, bkg_loc=bkg_loc)

    #step 5: superimpose at random locations (actual translations)
    location_x = [0, bkg.size[0]-logo.size[0]]
    location_y = [0, bkg.size[1]-logo.size[1]]

    loc = (np.random.randint(*location_x), np.random.randint(*location_y))

    bkg.paste(logo_tfms, loc, logo_tfms)
    
    if grayscale:
        bkg = transforms.Grayscale(num_output_channels=3)(bkg)
        #bkg = bkg.convert('LA')
        
    return bkg, (*size, *loc, loc[0] + logo_size[0], loc[1] + logo_size[1])

def visualize_bbox(img, bbox):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    
    draw.rectangle([(bbox[2],bbox[3]), (bbox[4],bbox[5])], outline=128)
    
    return img_copy

def generate_random_string(N=6):
    return ''.join([chr(i) for i in np.random.randint(97,122,size=N)])

def generate_augmented_dataset(N_images,
                               size=(512,512),
                               low_frac=0.30,
                               high_frac=0.90,
                               logo_file='CLOUDERALOGO.jpg',
                               N_tfms_to_apply=2,
                               seed=None,
                               save_loc=None,
                               logo_name=None,
                               generate=True,
                               bkg_loc=None,
                               perspective_tfm=False):
    
    if seed is not None: #should really figure out all torch seeds
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)    
    else:
        torch.backends.cudnn.deterministic = False
    
    img_list, bbox_list = [], []
    for i in range(N_images):
        img, bbox = generate_augmented_image(size, 
                                             logo_file, 
                                             N_tfms_to_apply,
                                             low_frac=low_frac,
                                             high_frac=high_frac,
                                             generate=generate,
                                             bkg_loc=bkg_loc,
                                             perspective_tfm=perspective_tfm)
        img_list.append(img)
        bbox_list.append(bbox)
        
    if save_loc is not None:
        if not os.path.exists(save_loc):
            pathlib.Path(save_loc).mkdir(parents=True, exist_ok=True)
        if logo_name is None:
            raise ValueError("Please enter logo_name if saving images")
                
        csv_fields = {'filename': [],
                      'width': [],
                      'height': [],
                      'class': [],
                      'xmin': [],
                      'ymin': [],
                      'xmax': [],
                      'ymax': []}
                
        for img, bbox in zip(img_list, bbox_list):
            img_name = f'{logo_name}_{generate_random_string()}.png'
            img.save(os.path.join(save_loc, img_name))
            
            csv_fields['filename'].append(os.path.join(save_loc, img_name))
            csv_fields['width'].append(bbox[0])
            csv_fields['height'].append(bbox[1])
            csv_fields['class'].append(logo_name)
            csv_fields['xmin'].append(bbox[2])
            csv_fields['ymin'].append(bbox[3])
            csv_fields['xmax'].append(bbox[4])
            csv_fields['ymax'].append(bbox[5])
            
        pd.DataFrame(csv_fields).to_csv(os.path.join(save_loc, f'{logo_name}.csv'), index=False)
    else:
        csv_fields = None
        
    return img_list, bbox_list, csv_fields

def convert_to_grayscale(color_loc, gray_loc):
    tfm = transforms.Grayscale(num_output_channels=3)
    
    if not os.path.exists(gray_loc):
        pathlib.Path(gray_loc).mkdir(parents=True, exist_ok=True)
        
    for img_file in glob.glob(os.path.join(color_loc, '*.png')):
        filename = img_file.split('/')[-1]
        
        img = Image.open(img_file).convert('LA')        
        img.save(os.path.join(gray_loc, filename))
        
    shutil.copy2(os.path.join(color_loc, 'rhlogo.csv'), os.path.join(gray_loc, 'rhlogo.csv'))
    shutil.copy2(os.path.join(color_loc, 'data.json'), os.path.join(gray_loc, 'data.json'))

    
