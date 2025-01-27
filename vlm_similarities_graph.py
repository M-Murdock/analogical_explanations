#!/usr/bin/env python3
# Reference: https://ai.google.dev/gemini-api/docs/quickstart?lang=python

# Similar to vlm_similarities, but creates a graph of relationships between trajectories
import google.generativeai as genai
import os
import PIL.Image
import re
import numpy as np
import json

# Feeds images of robot trajectories into the VLM. Assumes that we already have the images.
    
def main():
    genai.configure(api_key='AIzaSyC5ZYeBc-iVnqegb_X_EvIrc_Awa_6uCgk')
    # model = genai.GenerativeModel(model_name="gemini-1.5-flash") # garbage  
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

    # IMG_DIR = "trajectory_imgs/"
    # IMG_DIR = "icecream_traj_imgs/"
    IMG_DIR = "monocolor_icecream_traj_imgs/"
    
    # a list of paths to images
    img_paths = ["" for i in range(0, len(os.listdir(IMG_DIR)))]
    
    for file in os.listdir(IMG_DIR):
        if file.endswith(".png"):
            # make sure the list is ordered
            num = int(re.search(r"img(.*).png", file).group(1))
            img_paths[num] = os.path.join(IMG_DIR, file)

    # a list of images
    img_files = []
    for file in img_paths:
        img_files.append(PIL.Image.open(file))
        
    NUM_IMGS = len(img_files)
    NUM_IMGS = 10

    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    print("================================")
    # Build a database of images and their relationships to one another 
    # e.g. [{"A:pic1", "B:pic2", "relationship:shorter"}, {"A:", "B:", "relationship...""}]
    database = []

    # Get all the relationship prompts
    prompt = ""
    for i in range(0, NUM_IMGS):
        for j in range(0, NUM_IMGS):
            prompt += "What is the relationship between Trajectories {} and {}? Respond succinctly in a couple sentences but try to be descriptive. Do not repeat the prompt".format(i, j)
            prompt += "Separate each response with the character |. Try not to repeat a response more than a few times"
    
    # prompt gemini 
    img_files.insert(0, prompt)
    relationship_response = model.generate_content(img_files) 
    print(relationship_response.text)
    # print("relationship response: {}".format(relationship_response))
    relationship_list = relationship_response.text.split("|")
    # convert 1d array to 2d: [1,2,3,...] --> [[1,2,3],[4,5,6]]
    relationship_list = np.reshape(relationship_list, (-1, NUM_IMGS)).tolist()


    # create the database
    for i in range(0, NUM_IMGS):
        database.append([])
        for j in range(0, NUM_IMGS):
            database[-1].append({"relationship":relationship_list[i][j], "A":i, "B":j}) 


    # save database
    with open('database.json', 'w') as fout:
        json.dump(database, fout)
        
    
    # load database 
    with open("database.json", "r") as f:
        data = json.load(f)
        print(data)

    
    print("================================")

    
    
    
if __name__ == "__main__":
    main()