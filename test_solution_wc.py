import cv2
import time
import argparse
import mediapipe as mp 
import sys
import os
import pickle
import numpy as np
import facenet
import tensorflow.compat.v1 as tf
from scipy import misc 
from PIL import Image
def norm_to_pixels_ann(image,
                       relative_bounding_box,
                       score,
                       show_image= False):
    image_rows, image_cols, _ = image.shape
    denormalisation = mp.solutions.drawing_utils._normalized_to_pixel_coordinates
    rect_start_point = denormalisation(
        relative_bounding_box.xmin,
        relative_bounding_box.ymin,
        image_cols,
        image_rows)
    rect_end_point = denormalisation(
        relative_bounding_box.xmin + relative_bounding_box.width,
        relative_bounding_box.ymin + relative_bounding_box.height,
        image_cols,
        image_rows)
    bbox = [rect_start_point[0],
            rect_start_point[1],
            rect_end_point[0],
            rect_end_point[1],
            score[0]]
    if show_image == True:
        import matplotlib.pyplot as plt 
        plt.imshow(annotated_image)
        

    return bbox


def detect(src, 
           modeldir = './model/20170511-185253.pb',
           classifier_filename = './class/classifier.pkl',
           npy='./npy',
           train_img="./train_img"):

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            batch_size = 100
            image_size = 182
            input_image_size = 160
            
            HumanNames = os.listdir(train_img)
            HumanNames.sort()

            print('Loading Modal')
            facenet.load_model(modeldir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]


            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)


            cap= cv2.VideoCapture(src)
            past_time = 0
            face_detection = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5, model_selection=0)


            while True:
                _ , img = cap.read()
                #convert the images to grayScale
                #
                results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #run detection on rgb immages
                if results.detections:
                    bounding_boxes = np.array([
                        norm_to_pixels_ann(
                            img,
                            results.detections[detection_id].location_data.relative_bounding_box,
                            results.detections[detection_id].score,
                            show_image=False) for detection_id in range(len(results.detections))
                    ])
                    det = bounding_boxes[:, 0:4]
                    img_size = np.asarray(img.shape)[0:2]#image_height, image_width, _ = img.shape
                    cropped = []
                    scaled = []
                    scaled_reshape = []
                    bb = np.zeros((len(results.detections),4), dtype=np.int32)



                    for id, detection in enumerate(results.detections):
                        emb_array = np.zeros((1, embedding_size))
                        bb[id][0] = det[id][0]
                        bb[id][1] = det[id][1]
                        bb[id][2] = det[id][2]
                        bb[id][3] = det[id][3]

                        if bb[id][0] <= 0 or bb[id][1] <= 0 or bb[id][2] >= len(img[0]) or bb[id][3] >= len(img):
                            print('face is too close')
                            continue

                        cropped.append(img[bb[id][1]:bb[id][3], bb[id][0]:bb[id][2], :])
                        cropped[id] = facenet.flip(cropped[id], False)
                        scaled.append(np.array(Image.fromarray(cropped[id]).resize((image_size, image_size))))
                        scaled[id] = cv2.resize(scaled[id], (input_image_size,input_image_size),
                                            interpolation=cv2.INTER_CUBIC)
                        scaled[id] = facenet.prewhiten(scaled[id])
                        scaled_reshape.append(scaled[id].reshape(-1,input_image_size,input_image_size,3))
                        feed_dict = {images_placeholder: scaled_reshape[id], phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                        predictions = model.predict_proba(emb_array)
                        print(predictions)
                        best_class_indices = np.argmax(predictions, axis=1)
                        # print(best_class_indices)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        print(best_class_probabilities)
                        cv2.rectangle(img, (bb[id][0], bb[id][1]), (bb[id][2], bb[id][3]), (0, 255, 0), 2)    #boxing face

                        #plot result idx under box
                        text_x = bb[id][0]
                        text_y = bb[id][3] + 20
                        print('Result idx: ', best_class_indices[0])
                        for H_i in HumanNames:
                            # print(H_i)
                            if HumanNames[best_class_indices[0]] == H_i:
                                result_names = HumanNames[best_class_indices[0]]
                                cv2.putText(img, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)

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
    parser= argparse.ArgumentParser()
    parser.add_argument("--webcam",type=int, metavar='', required=False, help="The default pc Camera")
    args= parser.parse_args()
    detect(args.webcam)





