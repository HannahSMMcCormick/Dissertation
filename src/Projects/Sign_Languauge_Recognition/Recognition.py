#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2 
import os
import json
from config import MEDIAPIPE_PATH, EXTERNAL_PATH, INTERIM_PATH, hand_landmarker_task_PATH

#MediaPipe Classes
mp_hands = mp.tasks.vision.HandLandmarksConnections #Joints and fingers
mp_drawing = mp.tasks.vision.drawing_utils #Drawing on image
mp_drawing_styles = mp.tasks.vision.drawing_styles #Style of drawing on image

MARGIN = 10  #Margin of text from hand - Probs wont need this but could for testing
FONT_SIZE = 1 #Font formatting
FONT_THICKNESS = 1 #Font formatting 
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  #Colour


#Takes image and annotates it 
def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks #List of hands
  handedness_list = detection_result.handedness #Left or Right
  annotated_image = np.copy(rgb_image) #copy of image 

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx] #Joint landmarks
    handedness = handedness_list[idx] #Left or right

    # Draw the hand landmarks for each hand.
    mp_drawing.draw_landmarks(
      annotated_image, #Joint circles
      hand_landmarks, #Lines between
      mp_hands.HAND_CONNECTIONS, #Which joints match up
      mp_drawing_styles.get_default_hand_landmarks_style(), #Style of joints
      mp_drawing_styles.get_default_hand_connections_style()) #style of lines

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape #Get height and width of image
    x_coordinates = [landmark.x for landmark in hand_landmarks] #x-coordinate for all landmarks
    y_coordinates = [landmark.y for landmark in hand_landmarks] #y-coordinate for all landmarks
    text_x = int(min(x_coordinates) * width) #Place text at min x
    text_y = int(min(y_coordinates) * height) - MARGIN #Place text at min y

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def HandObject():
  base_options = python.BaseOptions(model_asset_path=str(hand_landmarker_task_PATH))
  options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2)
  detector = vision.HandLandmarker.create_from_options(options)
  
  for file, filename in enumerate(os.listdir(EXTERNAL_PATH)):
    
    input_file = os.path.join(EXTERNAL_PATH, filename)
    
    if not os.path.isfile(input_file):
      
      continue
    
    if not filename.lower().endswith(".mp4"):
      continue
    
    videoRead =  cv2.VideoCapture(input_file)
    
    if not videoRead.isOpened():
      print(f"Skipping (cannot open): {input_file}")
      continue
    
    fps = videoRead.get(cv2.CAP_PROP_FPS) or 30
    width = int(videoRead.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoRead.get(cv2.CAP_PROP_FRAME_HEIGHT))

    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    output_path = os.path.join(INTERIM_PATH, f"{base_name}_annotated.mp4") #Create Output dir
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
      
      ret, frame_bgr = videoRead.read()

      if not ret:
        break

      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
      
      detection_result = detector.detect(mp_image)
      
      annotated_rgb = draw_landmarks_on_image(frame_rgb, detection_result)
      annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

      writer.write(annotated_bgr)
      

    videoRead.release()
    writer.release()

if __name__ == "__main__":
    HandObject()





