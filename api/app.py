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

@app.post("/segmentation/image")
def image_segmentation(image: UploadFile=File(...)):
    array = read_image(image.file)
    image_shape = array.shape
    array = process_image(array)
    labels, centers = cluster(array)
    segmented_image = segment_image(image_shape, labels, centers)
    success, im = cv2.imencode('.png', segmented_image)
    headers = {'Content-Disposition': 'attachment; filename="test.tiff"'}
    return Response(im.tobytes() , headers=headers, media_type='image/tiff')

@app.post("/segmentation/matrix")
def segmentation_matrix(image: UploadFile=File(...)):
    array = read_image(image.file)
    image_shape = array.shape
    array = process_image(array)
    labels, centers = cluster(array)
    labeled_matrix = label_matrix(image_shape, labels, centers)
    stream = io.StringIO()
    labeled_matrix.to_csv(stream, ";", index=False)
    headers = {'Content-Disposition': 'attachment; filename="test.csv"'}
    return StreamingResponse(iter([stream.getvalue()]), 
                             headers=headers, 
                             media_type='text/csv')