"""

Script for saving the info of your members in a json file 

so that the identification process is carrying out smoothly

Format for running this script

there must be a folder - Database

Inside it 
each identity photo must be there and the photo named as <that_person_name>.{any_format /png/jpeg/jpg}

For example - shahrukh khan face photo saved with name shahrukh.jpg

Returns:
     a json file named as database.json

"""

import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from MTCNN import FaceDetector
import cv2
import json

database_folder  = "database" # path of that folder here

if __name__ == "__main__":
    mtcnn = MTCNN(select_largest = True , device = 'cuda') 
    # Create an inception resnet (in eval mode):
    reco = InceptionResnetV1(pretrained='vggface2').eval()
    for person in os.listdir(database_folder):
        name, extension = person.split(".")
        image = cv2.imread(os.path.join(database_folder, person))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fcd = FaceDetector(mtcnn, reco, is_image = True, img = image)
        embedding = fcd.run() 
        # adding peron by person data
        if not os.path.exists("Database.json"):  # if file not exist add the first person
            with open("Database.json", "w") as f:
                print(f"{name} - Data stored successfully")
                json.dump({name: embedding.numpy().tolist()}, f)
        else:
            with open("Database.json") as f:  # if exist the extract the original info and append the latest one
                database = json.load(f)
            database[name] = embedding.numpy().tolist()
            with open("Database.json", "w") as f:
                print(f"{name} - Data stored successfully")
                json.dump(database, f)
