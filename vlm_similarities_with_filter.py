#!/usr/bin/env python3
# Reference: https://ai.google.dev/gemini-api/docs/quickstart?lang=python

# Similar to vlm_similarities, but filters for 
import google.generativeai as genai
import os
import PIL.Image
import re
import numpy as np

# Feeds images of robot trajectories into the VLM. Assumes that we already have the images.
    
def main():
    genai.configure(api_key='AIzaSyC5ZYeBc-iVnqegb_X_EvIrc_Awa_6uCgk')
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    # IMG_DIR = "trajectory_imgs/"
    # IMG_DIR = "icecream_traj_imgs/"
    IMG_DIR = "monocolor_icecream_traj_imgs/"
    
    img_paths = ["" for i in range(0, len(os.listdir(IMG_DIR)))]
    
    for file in os.listdir(IMG_DIR):
        if file.endswith(".png"):
            num = int(re.search(r"img(.*).png", file).group(1))
            img_paths[num] = os.path.join(IMG_DIR, file)


    img_files = []
    for file in img_paths:
        img_files.append(PIL.Image.open(file))
        
        
    traj_num1 = 21
    traj_num2 = 24 
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    print("================================")
    # Rather than searching the entire set of trajectories, filter out those which seem irrelevant so we're just working with a subset
    
    # Heuristic: if there are a number of extremely similar trajectories, just keep one of them 
    prompt = "List a set of trajectories which are extremely similar to one another. Format your response as a list (e.g. [1,2,3,...])"
    img_files.insert(0, prompt)
    similarity_response = model.generate_content(img_files) 
    
    similar_trajs = re.search(r"\[(.*)\]",similarity_response.text) # list of trajectories which are extremely similar to one another
    similar_trajs_list = []
    if similar_trajs:
        similar_trajs_list = similar_trajs.group(1).split(",")


    # remove the similar trajectories
    traj_nums_to_remove = [t.strip() for t in similar_trajs_list if (int(t) != traj_num1) and (int(t) != traj_num2)]
    
    paths_to_remove = [os.path.join(IMG_DIR,"img{}.png".format(file_num)) for file_num in traj_nums_to_remove[1:]] # keep one of the trajectories
    print("PATHS TO REMOVE: ", paths_to_remove)   
    filtered_paths = [p for p in img_paths if p not in paths_to_remove]

    filtered_img_files = []
    for file in filtered_paths:
        filtered_img_files.append(PIL.Image.open(file))
    
    print("================================")

    # get the file names of the similar trajectories
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    
    traj1_index = filtered_paths.index(os.path.join(IMG_DIR,"img{}.png".format(traj_num1)))
    traj2_index = filtered_paths.index(os.path.join(IMG_DIR,"img{}.png".format(traj_num2)))

    pair1 = [filtered_img_files[traj1_index], filtered_img_files[traj2_index]]
    
    prompt = "What is the relationship between trajectories {} and {}?".format(traj_num1, traj_num2)
    prompt += " The trajectories are each drawn on the same graph, so pay attention to their relative sizes."
    
    pair1.insert(0, prompt)
    relationship = model.generate_content(pair1)
    print(relationship.text)
    
    print("================================")
    
    prompt = "You described the relationship between trajectories {} and {} using the following description. Find another pair of trajectory with this same relationship.".format(traj_num1, traj_num2)
    prompt += "Relationship: " + relationship.text
    filtered_img_files.insert(0, prompt)
    analogous_response = model.generate_content(filtered_img_files)
    print(analogous_response.text)
    
    
    
if __name__ == "__main__":
    main()