import cv2
import time
import argparse
import mediapipe as mp 
import sys



def detect():
    cap= cv2.VideoCapture(0)
    past_time = 0
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(0.5)


    while True:
        _ , img = cap.read()
        #convert the images to grayScale
        img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img_RGB)
        if results.detections:
            for id, detection in enumerate(results.detections):
                #print(id, detection) #return a dictionnary of detections found
                #print(detection.score)
                #print(detection.location_data.relative_bounding_box)
                normalized_bbox = detection.location_data.relative_bounding_box
                image_height, image_width, _ = img.shape

                x = int(normalized_bbox.xmin * image_width)
                y= int(normalized_bbox.ymin * image_height)
                w = int(normalized_bbox.width * image_width)
                h= int(normalized_bbox.height * image_height)
                
                bbox = x, y, w, h 
                x1, y1 = x + w, y + h
                l=30
                
                #top left thick rect
                cv2.line(img, (x,y) , (x+l, y),(255,0,255), 7) #corner thick line 
                cv2.line(img, (x,y) , (x, y+l),(255,0,255), 7) #corner thick line 
                #top right thick rect
                cv2.line(img, (x1,y) , (x1-l, y),(255,0,255), 7) #corner thick line 
                cv2.line(img, (x1,y) , (x1, y+l),(255,0,255), 7) #corner thick line 

                #bottom left thick rect
                cv2.line(img, (x,y1) , (x+l, y1),(255,0,255), 7) #corner thick line 
                cv2.line(img, (x,y1) , (x, y1-l),(255,0,255), 7) #corner thick line 
                #top right thick rect
                cv2.line(img, (x1,y1) , (x1-l, y1),(255,0,255), 7) #corner thick line 
                cv2.line(img, (x1,y1) , (x1, y1-l),(255,0,255), 7) #corner thick line 

                cv2.rectangle(img, bbox, (225,0,255),1)
                cv2.putText(img, 
                    f"{int(detection.score[0]*100)}%",
                    org= (bbox[0], bbox[1] - 20), 
                    fontFace= cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=2, 
                    color=(255,0,255),thickness=2)

        current_time = time.time()
        fps= 1/(current_time - past_time)
        print("detection with fps: ",fps)
        past_time = current_time
        cv2.putText(img, 
                    "Fps:{:.2f}".format(fps),
                    org=(20,70), 
                    fontFace= cv2.FONT_HERSHEY_PLAIN, 
                    fontScale=2, 
                    color=(0,255,0),thickness=2)
        cv2.imshow("Image", img)
        #cv2.waitKey(1) & 0xFF == ord('q')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    


if __name__=="__main__":
    detect()



