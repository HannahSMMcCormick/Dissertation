import os
import json
from config import EXTERNAL_PATH


videos = []

def Make_dataset():
    
    for file, filename in enumerate(os.listdir(EXTERNAL_PATH)):
      if filename.endswith(".mp4"):
        videos.append({
            "id": file + 1,
            "Sign": filename.replace(".mp4", ""),
            "filepath": os.path.join(EXTERNAL_PATH, filename),
            "HamNoSys": ""#Need to figure this out

        })
        
    data = {"videos": videos}
        
    with open("videos.json", "w") as f:
      json.dump(data, f, indent=4)

    print("All MP4 files added to JSON!")
    print(data)
    
        #add name of mp4 to json so I can pair with HAMNOSYS
        
Make_dataset()