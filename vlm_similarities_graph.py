#!/usr/bin/env python3
# Reference: https://ai.google.dev/gemini-api/docs/quickstart?lang=python

# Similar to vlm_similarities, but creates a graph of relationships between trajectories
import google.generativeai as genai
import os
import PIL.Image
import re
import numpy as np
import json
from enum import Enum

# Feeds images of robot trajectories into the VLM. Assumes that we already have the images.
    
def main():
    genai.configure(api_key='AIzaSyC5ZYeBc-iVnqegb_X_EvIrc_Awa_6uCgk')

    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

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
    # Build a tree of trajectories and their relationships to each other 

    # Get all the relationship prompts
    prompt = ""
    for i in range(0, NUM_IMGS):
        for j in range(0, NUM_IMGS):
            prompt += "What is the relationship between Trajectories {} and {}? Respond succinctly in a couple sentences but try to be descriptive. Do not repeat the prompt".format(i, j)
            prompt += "Separate each response with the character |. Try not to repeat a response more than a few times"
    
    # prompt gemini 
    img_files.insert(0, prompt)
    relationship_response = model.generate_content(img_files) 
    relationship_list = relationship_response.text.split("|")
    # # convert 1d array to 2d: [1,2,3,...] --> [[1,2,3],[4,5,6]]
    relationship_list = np.reshape(relationship_list, (-1, NUM_IMGS)).tolist()


    # make nodes for every image file
    nodes = [TrajNode(i) for i in range(0, NUM_IMGS)]

    
    # create the database
    for i in range(0, NUM_IMGS):
        for j in range(0, NUM_IMGS):
            nodes[i].add_edge(DirectedEdge(nodes[i], nodes[j], relationship_list[i][j]))
    nodes[0].edges[0].print_edge()

    # # save database
    # with open('database.json', 'w') as fout:
    #     json.dump(database, fout)
        
    
    # # load database 
    # with open("database.json", "r") as f:
    #     data = json.load(f)
    #     print(data)

    
    # print("================================")
    

    
# Edge directed away from a (a --> b) which indicates the "a is to b" relationship
class DirectedEdge():
    def __init__(self, a, b, r):
        self.a = a
        self.b = b
        self.r = r
    def print_edge(self):
        print("A: {}, B: {}, R: {}".format(self.a, self.b, self.r))

class TrajNode():
    def __init__(self, i):
        self.i = i
        # self.img = img
        self.edges = []
    def add_edge(self, e):
        self.edges.append(e)
    def print_node(self):
        print("i: {}, edges: {}".format(self.i, self.edges))
        
class R(Enum):
    SHORTER_THAN = 1
    LONGER_THAN = 2
    # ...
    
if __name__ == "__main__":
    main()