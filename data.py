from __future__ import print_function
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array,array_to_img
import numpy as np 
import os
import stat
import glob
import shutil
import skimage.io as io
import skimage.transform as trans
import pickle
#from PIL import image


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])

#base dict for augmentation parameters, can change here or also pass argument to function
data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        if len(mask.shape)==4:
            mask = mask[:,:,:,0]
        else:
            mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)


def trainGenerator_simple(batch_size,train_path,image_folder,mask_folder,aug_dict=data_gen_args,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class = False , num_class = 1)
        yield (img,mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict=data_gen_args,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,
                   target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict, validation_split=0.2, rescale=1./255)
    mask_datagen = ImageDataGenerator(**aug_dict, validation_split=0.2, rescale=1./255)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        subset = 'training',
        seed = seed) 
    image_val_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        subset = 'validation',
        seed = seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        subset = 'training',
        seed = seed)
    mask_val_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        subset = 'validation',
        seed = seed)
    
        
    return image_generator, image_val_generator, mask_generator, mask_val_generator

def train(image_generator, mask_generator, flag_multi_class = False, num_class = 2):
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
        
def val(image_val_generator, mask_val_generator, flag_multi_class = False, num_class = 2):
    val_generator = zip(image_val_generator, mask_val_generator)
    for (img_val,mask_val) in val_generator:
        img_val,mask_val = adjustData(img_val,mask_val,flag_multi_class,num_class)       
        yield (img_val,mask_val)
        
def test(test_path,target_size=(256,256)):
    X_test=[]
    names=[]
    for filename in os.listdir(test_path):
        name, ext = os.path.splitext(filename)
        if ext!=".png" and ext!=".jpg":
            continue
        names.append(filename)
        img=load_img(os.path.join(test_path,filename),target_size=target_size)
        img=img_to_array(img)/255
        X_test.append(img.copy())
    return np.array(X_test),names


def generate_labels(folder, names, num_img, size):
    test_label = np.zeros((num_img, size, size, 1))
    for (i, file) in enumerate(names):
        file = file.rstrip(".png")+".jpg"
        if file.startswith("2MSI"):
#             print(file)
            mask=load_img(os.path.join(folder,file),target_size=(size,size), color_mode="grayscale")
            mask=img_to_array(mask)
            if(np.max(mask) > 1):
                mask = mask /255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            test_label[i,:,:,:] = mask     
    return test_label

def generate_predict(folder, names, num_img, size, filenamestart):
    test_label = np.zeros((num_img, size, size, 1))
    for (i, file) in enumerate(names):
#         file = file.rstrip(".png")+".jpg"
        file = file.rstrip(".png")+"_predict.png"
#         print(filenamestart + file)
        mask=load_img(os.path.join(folder,filenamestart + file),target_size=(size,size), color_mode="grayscale")
        mask=img_to_array(mask)
        if(np.max(mask) > 1):
            mask = mask /255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            test_label[i,:,:,:] = mask
    return test_label
    
def create_pred(filestartname, model, folder_img = "dataset/test_images", folder_pred =  "/dataset/predict"):
    X_test,names=test(folder_img)
    names = [filestartname + name for name in names]
    preds=model_load.predict(X_test)
    preds=preds>0.5
    results=predToImgs(preds)
    saveResults(os.getcwd()+folder_pred,results,names,empty_dir=False)
    
def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)

def append_predict(filename):
    name, ext = os.path.splitext(filename)
    return "{name}_predict{ext}".format(name=name, ext=ext)

def predToImgs(results):
    results_img=np.zeros((results.shape[0],results.shape[1],results.shape[2],3))
    results_img[:,:,:,0]=results[:,:,:,0]
    results_img[:,:,:,1]=results[:,:,:,0]
    results_img[:,:,:,2]=results[:,:,:,0]
    results_img*=255
    return results_img


def saveResults(save_path,results,names,empty_dir=False):
    if empty_dir:
        clear_dir(save_path)
    results_img=predToImgs(results)
    for i in range(len(results)):
        img=array_to_img(results_img[i])
        img.save(os.path.join(save_path,append_predict(names[i])))
