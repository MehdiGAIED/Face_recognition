# demo_cerebrra
### Facenet: Real-time face recognition using deep learning Tensorflow 

This is completly based on deep learning nueral network and implented using Tensorflow framework. 

### Installation Python Libraries:
``` pip install requirements.txt ```
### Excution Face detection 

```python detect_face_mideapipe.py --webcam 0```

### Excution Face recognition 
The custom images dataset must be orgonized as follow inside the ```train_img``` folder :

train_img

    ├───Class1

        ├───img1.jpeg

        ├───img2.jpeg

        .

        .
        
        .

    ├───Class1

    ├───Class1

    ├───Class1



To excute the data preprocessing task Run : 
``` python data_preprocess.py ```


A novel dir ```pre_img``` will be added to the project that contains all extracted faces 

To start training the nn on the custom dataset run 
```python train_main.py```

Now a classifier is generated to classifie detected faces 


Now to run detection on images : 

Modifie the image path inside the  ```identify_face_image.py``` fole and run : 

```python identify_face_image.py```.

The saim thing remains True for the webcam and the videos detection. 

```python ideentify_face_video.py``` detect face within a video

```python identify_face_webcam.py``` detect face within webcam --> default webcam 0 


```python face_detection_facenet.py ``` present the same script for detecting images with image, video, or webcam but with local & global variables output 


The main cerebrra's improvements will be foccussing on the developpement and optimisation of ```python face_detection_mediapipe_image.py``` script with will combine the optimized running time of mediapipe and the CNN Facenet strength for detecting faces along with the classifier.pkl 
 

for compilation use ```python -m compileall -b <.py_FILE_NAME> ```