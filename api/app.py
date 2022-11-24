from fastapi import FastAPI, File, UploadFile

app = FastAPI()

@app.get("/image_segmentation/")
def segment_image(image: UploadFile=File(...)):
    res = {"message": "Hello world!"}
    return res
