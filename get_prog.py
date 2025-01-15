#!/usr/bin/env python3

import os

def main():
    IMG_DIR = "/Users/mavismurdock/Desktop/LfD_with_porgress/bags/"
    print("hello world!")
    img_paths = []
    for user_folder in os.listdir(IMG_DIR):
        print(user_folder)
        for file in os.listdir(user_folder):
            print(file)
        #     if file.endswith("_progress.txt"):
        #         img_paths.append(os.path.join(IMG_DIR, file))
                
    print(img_paths)
    # img_paths = []
    # for file in os.listdir(IMG_DIR):
    #     if file.endswith(".png"):
    #         img_paths.append(os.path.join(IMG_DIR, file))

    # img_files = []
    # for file in img_paths:
    #     img_files.append(PIL.Image.open(file))
    
if __name__ == "__main__":
    main()