import torch 
import pandas as pd 
import numpy as np 
from time import sleep
import requests
import subprocess

# import things for running autogen library 
import json
import os
import random
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import requests
from PIL import Image

# import autogen
# from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
# from autogen.agentchat.contrib.llava_agent import LLaVAAgent, llava_call
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import BitsAndBytesConfig
from transformers import pipeline
import torch

LLAVA_MODE = "local"  # Either "local" or "remote"
assert LLAVA_MODE in ["local", "remote"]

from IPython.display import display 
from IPython.display import Markdown

import PIL.Image

# Following is the code to use autogen 

# subprocess.Popen('python -m llava.serve.controller --host 0.0.0.0 --port 10000', shell=True)
# subprocess.Popen('python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:10000 --port 40000 --worker http://localhost:40000 --model-path liuhaotian/llava-v1.5-7b', shell=True)

# if LLAVA_MODE == "local":
#     llava_config_list = [
#         {
#             "model": "llava-v1.5-7b",
#             "api_key": "None",
#             "base_url": "http://0.0.0.0:10000",
#         }
#     ]

print("Loading the model: ")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

print("Model successfully loaded from pretrained!")

images = pd.read_csv("./data/annotations.csv")

# create a dictionary to store the uid, iid -> actual comment text, emotion 
comment_text_dict = dict() 

cnt = 0

# Autogen setup code, to make rest of code wait till controller and model worker are up: 
# print("Waiting till the model is running!")
# magic_phrase = "Application startup complete."

# with open("./controller.log") as f: 
#     while(True):
#         content = f.read()
#         if magic_phrase in content: 
#             print("Application startup complete found in controller!\n")
#             break

# while(True):
#     if not glob.glob("./model_worker*", recursive=False): 
#         continue 
#     model_worker = glob.glob("./model_worker*", recursive=False)[0]
#     with open(model_worker) as f: 
#         content = f.read()
#         if magic_phrase in content: 
#             print("Application startup complete found in Model worker!\n")
#             break

# iterate through the images that have features but no comments
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

# storing the map of label to idx 
label2idx = {
    "amusement": 0,
    "awe": 1,
    "contentment": 2,
    "excitement": 3,
    "anger": 4,
    "disgust": 5,
    "fear": 6,
    "sadness": 7 } 

labels = []
preds = [] 
pred_emotion_labels = []
print("Beginning the recognition process: ")
for _, row in images.iterrows():

    emotion_label = row['emotion']
    emotion_idx = label2idx[emotion_label]

    # derive the complete image path
    image_path = "/scratch/bbmr/sbhattacharyya1/projects-src/llm-eval/image/" + emotion_label + "/" + row['image_id'] + ".jpg"
    
    img = PIL.Image.open(image_path)

    prompt = "USER: <image>\nYou are a great bot with a good understanding of human emotions and feelings. Imagine you are shown an emotional photo, and based on that you have to answer what could possibly be evoked in a human when they view the image. Answer in a single word, choosing any one of the following emotion words: amusement, awe, contentment, excitement, anger, disgust, fear, sadness. Be sure to answer only in a single word, mentioning the emotion that is most likely to be evoked in a human when viewing the above given image.\nASSISTANT:"
        
    outputs = pipe(img, prompt=prompt, generate_kwargs={"max_new_tokens":5})
    comment = outputs[0]["generated_text"]
    asst_generated = comment.split("ASSISTANT")[1]
    asst_generated = asst_generated.strip()

    predicted_emotion_idx = label2idx[asst_generated.lower()]

    labels.append(emotion_idx)
    preds.append(predicted_emotion_idx)
    pred_emotion_labels.append(asst_generated.lower())
        
# check once all predictions are completed 
print(f"Total number of images in dataset = {len(images)}")        
print(f"Total number of generations created = {len(pred_emotion_labels)} = {len(preds)} = {len(labels)}")

# calculating the accuracy, f1, precision and recall 
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
print(f"Accuracy achieved = {accuracy_score(labels, preds)}")
print(f"Weighted precision = {precision_score(labels, preds, average='weighted')}")
print(f"Weighted recall = {recall_score(labels, preds, average='weighted')}")