import numpy as np
from PIL import Image
import pandas as pd
import cv2

NUMBER_OF_CLUSTERS = 4

def read_image(file):
    image = Image.open(file)
    array = np.array(image)
    return array

def process_image(image):
    array = image.reshape((-1, 4))
    array = np.float32(array)
    array = pd.DataFrame(array).drop(3, axis=1)
    array = array.to_numpy()
    return array

def cluster(image, max_iterations: int=100,  epsilon: float=1.0):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                max_iterations, epsilon)
    compactness, labels, centers = cv2.kmeans(image, NUMBER_OF_CLUSTERS, 
                                              None, criteria, 10, 
                                              cv2.KMEANS_RANDOM_CENTERS)
    return labels, centers

def segment_image(image_shape, labels, centers):
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image_shape[0], 
                                              image_shape[1], 
                                              3))
    return segmented_image

    
    

