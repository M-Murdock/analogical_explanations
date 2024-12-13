#!/usr/bin/env python3

import google.generativeai as genai
import httpx
import os
import base64
import PIL.Image


    
def main():
    genai.configure(api_key='AIzaSyDRxCLCKF95i2leizDmc5K23sxm4MWmqNc')

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")

    IMG_DIR = "trajectory_imgs/"
    
    image_path_1 = os.path.join(IMG_DIR, "photo2.png")   # Replace with the actual path to your first image
    image_path_2 = os.path.join(IMG_DIR, "photo3.png")  # Replace with the actual path to your second image

    sample_file_1 = PIL.Image.open(image_path_1)
    sample_file_2 = PIL.Image.open(image_path_2)

    prompt = "Compare the dotted paths in these images"

    response = model.generate_content([prompt, sample_file_1, sample_file_2])

    print(response.text)
    
if __name__ == "__main__":
    main()