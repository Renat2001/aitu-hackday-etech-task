import cv2
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response, StreamingResponse
from ai_daniyar.utils import *

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.post("/segmentation/matrix")
def image_segmentation(image: UploadFile=File(...)):
    save_file(UPLOAD_FILE_PATH, image)
    filepath = os.path.join(UPLOAD_FILE_PATH, image.filename)
    data = read_image(filepath)
    ndvi = es.normalized_diff(data[3], data[0])
    flat_data = process_image(ndvi)
    model = get_model("C:/Users/Acer/Desktop/github/aitu_hackday_rid/jupyters/rid_model.pkl")
    predicted_data = cluster(model, flat_data)
    matrix = create_matrix(predicted_data, ndvi)
    return matrix
    # array = read_image(image.file)
    # image_shape = array.shape
    # array = process_image(array)
    # labels, centers = cluster(array)
    # segmented_image = segment_image(image_shape, labels, centers)
    # success, im = cv2.imencode('.png', segmented_image)
    # headers = {'Content-Disposition': 'inline; filename="test.tiff"'}
    # return Response(im.tobytes() , headers=headers, media_type='image/tiff')

@app.post("/segmentation/image")
def segmentation_matrix(image: UploadFile=File(...)):
    save_file(UPLOAD_FILE_PATH, image)
    filepath = os.path.join(UPLOAD_FILE_PATH, image.filename)
    data = read_image(filepath)
    ndvi = es.normalized_diff(data[3], data[0])
    flat_data = process_image(ndvi)
    model = get_model("C:/Users/Acer/Desktop/github/aitu_hackday_rid/jupyters/rid_model.pkl")
    predicted_data = cluster(model, flat_data)
    matrix = create_matrix(predicted_data, ndvi)
    return matrix