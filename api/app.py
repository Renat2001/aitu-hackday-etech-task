import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
from ai_daniyar.utils import *
from PIL import Image

app = FastAPI()

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])

@app.post("/image_segmentation/")
def image_segmentation(image: UploadFile=File(...)):
    array = read_image(image.file)
    image_shape = array.shape
    array = process_image(array)
    labels, centers = cluster(array)
    segmented_image = segment_image(image_shape, labels, centers)
    success, im = cv2.imencode('.png', segmented_image)
    headers = {'Content-Disposition': 'attachment; filename="test.tiff"'}
    return Response(im.tobytes() , headers=headers, media_type='image/tiff')

