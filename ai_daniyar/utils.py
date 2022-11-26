import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import pickle
import pandas as pd
import shutil
import os
import matplotlib.colors as mc

UPLOAD_FILE_PATH = './static/images/'
NUMBER_OF_CLUSTERS = 4

def save_file(path, file):
  file_object = file.file
  upload_folder = open(os.path.join(path, file.filename), 'wb+')
  shutil.copyfileobj(file_object, upload_folder)
  upload_folder.close()

def read_image(file):
    data = rxr.open_rasterio(file)
    return data

def process_image(ndvi):
    data = ndvi.data
    data = np.nan_to_num(data, nan=-1)
    flat_data = data[:,:].reshape((data.shape[0]*data.shape[1], 1))
    return flat_data

def cluster(model, flat_data):
    centers = sorted(model.cluster_centers_.tolist())
    grey_range = [-1, centers[1]]
    red_range = [centers[1][0], centers[2][0]]
    yellow_range = [centers[2][0], centers[3][0]]
    green_range = [centers[3][0], 1]
    flat_data[(flat_data>=grey_range[0]) 
               & (flat_data<grey_range[1])] = 1
    flat_data[(flat_data>=red_range[0]) 
               & (flat_data<red_range[1])] = 2
    flat_data[(flat_data>=yellow_range[0]) 
               & (flat_data<yellow_range[1])] = 3
    flat_data[(flat_data>=green_range[0]) 
               & (flat_data<green_range[1])] = 4
    return flat_data

def get_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

def create_matrix(predicted_data, data):
    matrix = predicted_data.reshape(data[:,:].shape)
    matrix = pd.DataFrame(matrix).to_dict()
    return matrix

def segment_image(image_shape, labels, centers):
    # Create a custom color map to represent our different 4 classes
    cmap = mc.LinearSegmentedColormap.from_list("", ["grey","yellow","red","green"])
    # Show the resulting array and save it as jpg image
    plt.figure(figsize=[20,20])
    plt.imshow(elman_cul, cmap=cmap)
    plt.axis('off')
    plt.savefig("elhas_clustered.jpg", bbox_inches='tight')
    plt.show()

def label_matrix(image_shape, labels, centers):
    centers = np.uint8(centers)
    labeled_matrix = labels.reshape((image_shape[0], 
                                     image_shape[1]))
    labeled_matrix[labeled_matrix==0] = 4
    labeled_matrix = pd.DataFrame(labeled_matrix)
    return labeled_matrix

    
    

