#!/usr/bin/env python3
# Reference: https://ai.google.dev/gemini-api/docs/quickstart?lang=python

import google.generativeai as genai
import os
import PIL.Image

# Feeds images of robot trajectories into the VLM. Assumes that we already have the images.
    
def main():
    genai.configure(api_key='AIzaSyC5ZYeBc-iVnqegb_X_EvIrc_Awa_6uCgk')
    model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp")

    # IMG_DIR = "trajectory_imgs/"
    # IMG_DIR = "icecream_traj_imgs/"
    IMG_DIR = "monocolor_icecream_traj_imgs/"
    
    img_paths = []
    for file in os.listdir(IMG_DIR):
        if file.endswith(".png"):
            img_paths.append(os.path.join(IMG_DIR, file))

    img_files = []
    for file in img_paths:
        img_files.append(PIL.Image.open(file))
        

    # prompt = "Each trajectory begins at the red dot and ends at the green dot. The trajectories are all shown at the same scale. All trajectories end at the same place, so ignore this end state when comparing trajectories."
    # prompt = "These images are all trajectories. Each begins at the red dot and ends at the green dot. The trajectories are each drawn on the same graph. They all end in the same location."
    # prompt = "These images are all trajectories. The shapes of trajectories 46 and 47 share a relationship. What other trajectory shapes share the same relationship?""
    # prompt += "Trajectories 46 and 47 share a relationship. What is another pair of trajectory shapes that share the same relationship? This pair will be analogous to trajectories 46 and 47."
    
    prompt = "These images are all trajectories. The trajectories are each drawn on the same graph, so pay attention to their relative sizes."
    prompt += "What is the relationship between the shapes of trajectories 1 and 2? What are two other trajectories which share this relationship?"

    # prompt += "What is the confidence of your answer?"
    img_files.insert(0, prompt)
    

    response = model.generate_content(img_files)

    print(response.text)
    
if __name__ == "__main__":
    main()