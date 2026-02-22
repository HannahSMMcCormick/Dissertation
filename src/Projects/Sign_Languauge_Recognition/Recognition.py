import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles
import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os
from config import EXTERNAL_PATH, INTERIM_PATH, hand_landmarker_task_PATH, Face_Landmarker_task_PATH, pose_landmarker_task_PATH
import json

HAND_LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC","THUMB_MCP","THUMB_IP","THUMB_TIP",
    "INDEX_FINGER_MCP","INDEX_FINGER_PIP","INDEX_FINGER_DIP","INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP","MIDDLE_FINGER_PIP","MIDDLE_FINGER_DIP","MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP","RING_FINGER_PIP","RING_FINGER_DIP","RING_FINGER_TIP",
    "PINKY_MCP","PINKY_PIP","PINKY_DIP","PINKY_TIP"
]

#MediaPipe Classes
mp_hands = mp.tasks.vision.HandLandmarksConnections #Joints and fingers
mp_drawing = mp.tasks.vision.drawing_utils #Drawing on image
mp_drawing_styles = mp.tasks.vision.drawing_styles #Style of drawing on image

MARGIN = 10  #Margin of text from hand - Probs wont need this but could for testing
FONT_SIZE = 1 #Font formatting
FONT_THICKNESS = 1 #Font formatting 
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  #Colour


#Takes image and annotates it 
def draw_landmarks_on_image(rgb_image, hand_detection_result, face_detection_result, pose_detection_result):
  
  #Hand Landmarks
  hand_landmarks_list = hand_detection_result.hand_landmarks #List of hands
  handedness_list = hand_detection_result.handedness #Left or Right
  

  # Face Landmarks
  face_landmarks_list = face_detection_result.face_landmarks 
  
  #Pose Landmarks
  pose_landmarks_list = pose_detection_result.pose_landmarks #check dependencies
  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
  
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

  for idx in range(len(face_landmarks_list)):
    
    face_landmarks = face_landmarks_list[idx]
    
    
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks,
        connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
          landmark_drawing_spec=None,
          connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style())
  
  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)
    
  

  return annotated_image



def HandObject():
  
  #Create HandObject
  base_options = python.BaseOptions(model_asset_path=str(hand_landmarker_task_PATH))
  options = vision.HandLandmarkerOptions(base_options=base_options,num_hands=2)
  detector = vision.HandLandmarker.create_from_options(options)
  
  return detector

def FaceObject():

  base_options = python.BaseOptions(model_asset_path= str(Face_Landmarker_task_PATH))
  options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
  detector = vision.FaceLandmarker.create_from_options(options)
  
  
  return detector
  
def poseObject():
  
  base_options = python.BaseOptions(model_asset_path= str(pose_landmarker_task_PATH))
  options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
  detector = vision.PoseLandmarker.create_from_options(options)
  
  return detector

def get_video():
  
  #Loop through all videos
  for _, filename in enumerate(os.listdir(EXTERNAL_PATH)):
    
    
    input_file = os.path.join(EXTERNAL_PATH, filename) #Input file path
    
    if not os.path.isfile(input_file): #If file doesn't exist at oath, skip
      
      continue
    
    if not filename.lower().endswith(".mp4"): #If file isn't video, skip
      continue
    
    videoRead =  cv2.VideoCapture(input_file) #Read video
    
    if not videoRead.isOpened(): #If video can't open the skip
      print(f"Skipping (cannot open): {input_file}")
      continue
    
    fps = videoRead.get(cv2.CAP_PROP_FPS) #Get frames per second from video or set it to 30
    
    width = int(videoRead.get(cv2.CAP_PROP_FRAME_WIDTH)) #Get Width of video
    height = int(videoRead.get(cv2.CAP_PROP_FRAME_HEIGHT)) #Get Height of video

    base_name = os.path.splitext(os.path.basename(filename))[0] #Take basename(Sign in video) to use for output video
    
    output_path = os.path.join(INTERIM_PATH, f"{base_name}_annotated.mp4") #Create Output dir
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") #Compress code for output
    
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) #Write Output video

    #Json format for extracted points (so far only for hand landmarks, will add face and pose later)
    json_output = {
    "video_path": input_file,
    "fps": fps,
    "frames": []
    }
    
    hand_landmark_detector = HandObject()
    
    face_landmark_detector = FaceObject()
    
    pose_landmark_detector = poseObject()
    
    frame_idx = 0
    
    while True:
      
      ret, frame_bgr = videoRead.read() #Does frame exist and if so save it

      #If reached end of video the break
      if not ret:
        break


      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) #Convert video from bgr to rgb
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb) #Makes numpy array compatible with mediapipe
      
      hand_detection_result = hand_landmark_detector.detect(mp_image) #Detects hand landmarks
      face_detection_result = face_landmark_detector.detect(mp_image) #Detects face landmarks
      pose_detection_result = pose_landmark_detector.detect(mp_image)
      
      annotated_rgb = draw_landmarks_on_image(frame_rgb, hand_detection_result, face_detection_result, pose_detection_result) #Use the function to draw the landmarks on
      annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR) #convert back to bgr
  
      writer.write(annotated_bgr)#Makes new video
      
      #extract points to put in JSON
      json_output["frames"].append(
        Extract_points(hand_detection_result, frame_idx, fps)
      )

      #Increment frame
      frame_idx += 1
      
      
    #Currentlt putting JSON in interim folder with videos, might change later
    json_path = os.path.join(INTERIM_PATH, f"{base_name}_hands.json")
    with open(json_path, "w", encoding="utf-8") as f:
      json.dump(json_output, f, indent=2)

    print(f"Saved hand landmarks JSON: {json_path}")
          #Closes everything
    videoRead.release()
    writer.release()
      
def Extract_points(hand_detection_result, frame_idx, fps):
  
  #Json of frame index, when that frame is and which hand the points are coming from
    frame_record = {
        "frame_index": frame_idx,
        "time_sec": frame_idx / fps,
        "hands": []
    }
      
    
    hand_landmarks_list = hand_detection_result.hand_landmarks or [] # Does hands exist
    handedness_list = hand_detection_result.handedness or [] # Does handidness exist
    
    
    for h_i, hand_landmarks in enumerate(hand_landmarks_list):
        # handedness for this hand 
        label = None
        score = None
        if h_i < len(handedness_list) and handedness_list[h_i]:
            
            catagories = handedness_list[h_i][0]  #list of catagories
            label = catagories.category_name #left/right
            score = float(catagories.score) #Models confidence in choosing left/right hand correctly

        points = [] #landmarks
        for lm_i, lm in enumerate(hand_landmarks):
            points.append({
                "id": lm_i,
                "name": HAND_LANDMARK_NAMES[lm_i],
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
            })

        # add hand to frame  
        frame_record["hands"].append({
            "hand_index": h_i,
            "handedness": label,
            "handedness_score": score,
            "landmarks": points
        })

    return frame_record

"""
  change structure 
  One function to loop videos + get specs
  That function calls head object and face object
  then first function calls draw function
  
  -Done
"""



#Run program
if __name__ == "__main__":
    get_video()





