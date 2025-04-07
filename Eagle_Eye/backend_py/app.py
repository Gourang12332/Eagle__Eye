import cv2
import numpy as np
from deepface import DeepFace
import torch
from ultralytics import YOLO
import cv2
import nest_asyncio
import uvicorn
import requests
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from io import BytesIO
from pyngrok import ngrok
from torchvision import transforms
from PIL import Image
import sys
import json
import faiss
import tempfile
from typing import List
import open_clip
import uuid
from pydantic import BaseModel
import cloudinary
import cloudinary.uploader
import cloudinary.api
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
sys.path.append("/content/fast-reid/")
from fastreid.config import get_cfg
from fastreid.engine import DefaultPredictor
print("Torch CUDA Available:", torch.cuda.is_available())
print("Torch Version:", torch.__version__)
print("CUDA Version:", torch.version.cuda)
yolo_model=YOLO("yolov8n.pt")

def extract_bounding_box(image_path):
  img=cv2.imread(image_path)
  print(img)
  results=yolo_model(img)
  detections=results[0].boxes
  bound_box=detections.xyxy.cpu().numpy()
  conf_score=detections.conf.cpu().numpy()
  classes=detections.cls.cpu().numpy()
  return bound_box,conf_score,classes,img

def crop_image(bound_box,conf_score,classes,image):
  cropped_images=[]
  for box,conf,class_id in zip(bound_box,conf_score,classes):
    if class_id==0 and conf>0.5:
      x1, y1, x2, y2 = map(int, box)
      cropped_img = image[y1:y2, x1:x2]
      cropped_images.append(cropped_img)
  return cropped_images

def preprocess_image(cropped_images):
  transform = transforms.Compose([
    transforms.Resize((256, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  preprocessed_image=[transform(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))) for img in cropped_images]
  return preprocessed_image

def get_embeddings(preprocessed_image):
  if not preprocessed_image :
    return None
  cfg = get_cfg()
  cfg.merge_from_file("/content/fast-reid/configs/Market1501/bagtricks_R50.yml")
  cfg.MODEL.WEIGHTS = "/content/market_bot_R50.pth"
  cfg.MODEL.DEVICE= "cuda"
  print(cfg.MODEL.DEVICE)
  predictor = DefaultPredictor(cfg)
  preprocessed_images_tensor = torch.stack(preprocessed_image).to("cuda")
  with torch.no_grad():
    embeddings = predictor(preprocessed_images_tensor)
  print(embeddings.shape)
  return embeddings


def get_metadata_for_embedding(query_embedding, faiss_index, metadata_file):
    with open(metadata_file, "r") as f:
        metadata_store = json.load(f)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    print(query_embedding.shape)
    query_embedding=query_embedding[0]
    query_embedding /= np.linalg.norm(query_embedding)
    print(query_embedding)
    D, I = faiss_index.search(query_embedding.reshape(1, -1), 1)

    confidence = 1 / (1 + D[0][0])

    best_match_id = I[0][0]

    if best_match_id == -1:
        return {"error": "No matching embedding found"}
    best_match_id_str = str(best_match_id)
    metadata = metadata_store.get(best_match_id_str, {"error": "ID not found in metadata"})
    metadata["confidence"]=confidence

    
    return metadata

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class CCTVData(BaseModel):
    timestamp: str
    location: str
    latlong: List[float]
    CCTV_id: int
    Url: str

def video_embeddings(data:CCTVData):
  output_folder = "/content/extracted_frames"
  os.makedirs(output_folder, exist_ok=True)
  def download_video(url):
    temp_dir = "/content/"
    os.makedirs(temp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".mp4") as temp_video:
        response = requests.get(url, stream=True)
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            temp_video.write(chunk)
        return temp_video.name

  video_path = download_video(data.Url)
  print(f"Downloaded Video Path: {video_path}")
#  Gourang Jian
  cap = cv2.VideoCapture(video_path)

  fps = cap.get(cv2.CAP_PROP_FPS)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  duration_sec = int(total_frames / fps)

  frame_interval = int(fps)
  frame_count = 0
  saved_frame_count = 0

  while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
          break

      if frame_count % frame_interval == 0:
          frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count}.jpg")
          cv2.imwrite(frame_filename, frame)
          saved_frame_count += 1

      frame_count += 1

  cap.release()
  print(f"Saved {saved_frame_count} frames to '{output_folder}'")
  embeddings=process_all_frames(output_folder)
  print(embeddings)

  index_path = "/content/demo_faiss.index"
  metadata_path = "/content/demo_metadata.json"

  if os.path.exists(index_path):
      index = faiss.read_index(index_path)
      print("Loaded existing FAISS index.")
  else:
      index = faiss.IndexFlatIP(2048)
      index = faiss.IndexIDMap(index)
      print("Initialized new FAISS index (empty).")


  if os.path.exists(metadata_path):
      with open(metadata_path, "r") as f:
          metadata_store = json.load(f)
      print("Loaded existing metadata store.")
  else:
      metadata_store = {}
      print("Initialized new metadata store (empty).")

  final_embeddings = []

  for emb in embeddings:
        emb = np.array(emb, dtype=np.float32)
        if len(emb.shape) > 1:
            for face_emb in emb:
                final_embeddings.append(face_emb)
        else:
            final_embeddings.append(emb)
  print(final_embeddings)
  final_embeddings=np.array(final_embeddings)
  video_metadata = {
      "timestamp": data.timestamp,
      "location": data.location,
      "latlong": data.latlong,
      "camera_id": data.CCTV_id
  }


  for emb in final_embeddings:
      if index.ntotal == 0:

          new_id = uuid.uuid4().int >> 64
          result=index.add_with_ids(np.array([emb]), np.array([new_id]))
          print(result)
          metadata_store[str(new_id)] = video_metadata.copy()
      else:
          D, I = index.search(np.array([emb]), k=1)
          top_score = D[0][0]
          top_id = I[0][0]
          if top_score >= 0.999:
              matched_meta = metadata_store.get(str(top_id), {})
              if matched_meta and matched_meta.get("location") == video_metadata["location"]:
                  matched_meta["timestamp"] = video_metadata["timestamp"]
                  metadata_store[str(top_id)] = matched_meta
              else:
                  new_id = uuid.uuid4().int >> 64
                  index.add_with_ids(np.array([emb]), np.array([new_id]))
                  metadata_store[str(new_id)] = video_metadata.copy()



      faiss.write_index(index, index_path)
      with open(metadata_path, "w") as f:
          json.dump(metadata_store, f)

  print("Processed embeddings, saved index and metadata.")
  print(f"Total vectors in FAISS index: {index.ntotal}")

import torch

def process_description(description):
    model, preprocess, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    text = tokenizer([description])

    with torch.no_grad():
        text_embedding = model.encode_text(text)  
    
    
    text_embedding = text_embedding.cpu().numpy()
    padded_embedding = np.pad(text_embedding, ((0, 0), (0, 1536)), mode='constant')  # Pad with zeros
    
    return padded_embedding

frame_folder = "/content/extracted_frames"
def process_all_frames(folder):
    embeddings_list = []

    for filename in sorted(os.listdir(folder)):
        image_path = os.path.join(folder, filename)
        print(f"Processing: {image_path}")
        bound_box, conf_score, classes, img = extract_bounding_box(image_path)
        cropped_images = crop_image(bound_box, conf_score, classes, img)
        if not cropped_images:
            print(f"No valid objects detected in {filename}")
            continue
        preprocessed_images = preprocess_image(cropped_images)

        embeddings = get_embeddings(preprocessed_images)
        print(embeddings.shape)
        if embeddings is not None:
            embeddings_list.append(embeddings.cpu().numpy())

    return embeddings_list

all_embeddings = process_all_frames(frame_folder)
print(f"Total frames processed: {len(all_embeddings)}")
print(all_embeddings)

cloudinary.config(
    cloud_name='YOUR_CLOUD_NAME',
    api_key='YOUR_API_KEY',
    api_secret='YOUR_API_SECRET'
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class CCTVData(BaseModel):
    timestamp: str
    location: str
    latlong: List[float]
    CCTV_id: str
    Url: str

@app.get("/")
def system_check():
    return {"message": "FastAPI server is running actively on Google Colab!"}

class VideoRequest(BaseModel):
    video_url: str
    metadata: dict

class DescriptionRequest(BaseModel):
    description: str

@app.post("/video-count-population")
async def count_population(request: VideoRequest):
    try:
        video_url = request.video_url
        metadata = request.metadata
        response = requests.get(video_url, stream=True)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download video")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                temp_video.write(chunk)
            temp_video_path = temp_video.name
        timeline_data = process_video(temp_video_path)

        return {
            "video_url": video_url,
            "timeline_data": timeline_data,
            "metadata": metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = fps * 2
    frame_count = 0
    timeline_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            second = (frame_count // fps)
            results = yolo_model(frame)
            people_count = sum(1 for obj in results[0].boxes.cls if int(obj) == 0)
            timeline_data.append({
                "second": second,
                "max_people_found": people_count
            })

            print(f"At {second}s: {people_count} people detected.")

        frame_count += 1

    cap.release()
    return timeline_data

@app.post("/cctv-data")
async def receive_cctv_data(data: CCTVData):
    print(data.Url)
    video_embeddings(data)
    return {"message": "Data received successfully", "data": data}


@app.post("/photo-predict")
async def predict_person(image_file: UploadFile = File(...)):
    print(f"Received file: {image_file.filename}, Content Type: {image_file.content_type}")

    try:
        content = await image_file.read()
        upload_response = cloudinary.uploader.upload(content)

        image_url = upload_response['secure_url']
        print(f"Image uploaded to Cloudinary. URL: {image_url}")
        resp = requests.get(image_url)
        image_array = np.frombuffer(resp.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file"})
        image_path = "frame_-1.jpg"
        cv2.imwrite(image_path, image)
        bound_box, conf_score, classes, image = extract_bounding_box(image_path)
        cropped_images = crop_image(bound_box, conf_score, classes, image)
        preprocessed_image = preprocess_image(cropped_images)
        embeddings = get_embeddings(preprocessed_image)
        metadata=get_metadata_for_embedding(embeddings,faiss_index=faiss.read_index("/content/demo_faiss.index"),metadata_file = "/content/demo_metadata.json")
        if embeddings is not None and embeddings.numel() > 0:
            return  JSONResponse(status_code=200,content={"metadata": metadata})
        else:
            return JSONResponse(status_code=400,content={"message": "No similar person found in the database."})

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.post("/description")
def description(data: DescriptionRequest):
  try:
    embeddings=process_description(data.description)
    print(embeddings)
    metadata=get_metadata_for_embedding(embeddings,faiss_index=faiss.read_index("/content/demo_faiss.index"),metadata_file = "/content/demo_metadata.json")
    if embeddings is not None and embeddings.size > 0:
        return  JSONResponse(status_code=200,content={"metadata": metadata})
    else:
        return JSONResponse(status_code=400,content={"message": "No similar person found in the database."})

  except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


def start_ngrok():
    ngrok.kill()
    ngrok.set_auth_token("YOUR_AUTH_TOKEN")
    public_url = ngrok.connect(8000).public_url
    print(f"Public URL: {public_url}")
    return public_url

def start_server():
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)

public_url = start_ngrok()
start_server()