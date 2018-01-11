## Project: Vehicle Detection and Tracking
### Dependencies:
1. Python 3.6
2. OpenCV
3. Pickle
4. Numpy
5. Matplotlib
6. Glob
7. Time
8. Scikit learn

---
**Folders in Directory:**


    1. output_images: Contains output of pipeline on test images with heatmap treshold of 1
    2. test_images: Contains test images for this project
    3. writeup_images: Contains images included in the writeup
    4. Model: Contains a trained SVM model for classification of vehicles and non-vehicles
    
---
**File in Directory:**

1. `writeup.md`
2. _`HOG.ipynb`_ : Contains code for training the SVM model and saving it.
3. _`Sliding Window Search and Video.ipynb`_ : Contains code for vehicle detection in an image, multiple images and video using sliding window technique and the trained classifier.
4. *`project_video_output.ipynb`* : Video file of the output.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_images/data_visualize.jpg
[image2]: ./writeup_images/hog.jpg

[image3]: ./writeup_images/first_layer.jpg
[image4]: ./writeup_images/second_layer.jpg
[image5]: ./writeup_images/third_layer.jpg
[image6]: ./writeup_images/combined.jpg


[image7]: ./writeup_images/hog-sub.jpg

[image8]: ./writeup_images/first.jpg
[image9]: ./writeup_images/second.jpg
[image10]: ./writeup_images/third.jpg
[image11]: ./writeup_images/combined.jpg

[image12]: ./writeup_images/heatmap1.jpg
[image13]: ./writeup_images/heatmap2.jpg
[image14]: ./writeup_images/heatmap3.jpg
[image15]: ./writeup_images/heatmap4.jpg
[image16]: ./writeup_images/heatmap5.jpg
[image17]: ./writeup_images/heatmap6.jpg

[image18]: ./writeup_images/label1.jpg
[image19]: ./writeup_images/label2.jpg
[image20]: ./writeup_images/label3.jpg
[image21]: ./writeup_images/label4.jpg
[image22]: ./writeup_images/label5.jpg
[image23]: ./writeup_images/label6.jpg

[image24]: ./writeup_images/result.jpg

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### 1. Writeup: 
 Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. Here is a template writeup for this project you can use as a guide and a starting point.

### 2. HOG (Histogram of Oriented Gradients):
1. Explain how (and identify where in your code) you extracted HOG features from the training images.
2. Explain how you settled on your final choice of HOG parameters.
3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

### 3. Sliding Window Search:
1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?
2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

### 4. Video Implementation:
1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

### 5. Discussion:

---
### I. Writeup

#### Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

`writeup.md` in the directory is the file explaining the project.

---

### II. Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

- The code for this step is contained in the 1st to 19th code cell of the IPython notebook (`HOG_with_color_features.ipynb`).  

- I started by reading in all the `vehicle` and `non-vehicle` images. Code for this is available in 2nd and 3rd cell. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

- I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

- Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

- The code for above output is available in code cell in 7th and 8th. Function used for this is `get_hog_features()`

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters:
|**Color-space**|**Hog Channel**|**Training Time**|**Accuracy** |
| ------------ | ------------- | ---------------- | ----------- |
|     RGB      |       0       |    11.23 sec     |  94.12 %    |
|     RGB      |       1       |    07.97 sec     |  95.43 %    |
|     RGB      |       2       |    08.55 sec     |  94.87 %    |
|     RGB      |      ALL      |    90.86 sec     |  97.10 %    |
|||||
|     LUV      |       0       |    07.23 sec     |  95.36 %    |
|     LUV      |       1       |    05.85 sec     |  95.68 %    |
|     LUV      |       2       |    10.58 sec     |  94.02 %    |
|     LUV      |      ALL      |    143.00 sec    |  99.11 %    |
|||||
|     HSV      |       0       |    08.33 sec     |  95.54 %    |
|     HSV      |       1       |    14.14 sec     |  92.39 %    |
|     HSV      |       2       |    08.03 sec     |  94.16 %    |
|     HSV      |      ALL      |    98.10 sec     |  98.76 %    |
|||||
|     HLS      |       0       |    08.89 sec     |  95.82 %    |
|     HLS      |       1       |    07.59 sec     |  95.08 %    |
|     HLS      |       2       |    14.37 sec     |  91.11 %    |
|     HLS      |      ALL      |   118.87 sec     |  99.36 %    |
|||||
|     YUV      |       0       |    08.19 sec     |  95.43 %    |
|     YUV      |       1       |    10.23 sec     |  94.02 %    |
|     YUV      |       2       |    05.25 sec     |  95.79 %    |
|     YUV      |      ALL      |    17.76 sec     |  99.54 %    |
|||||
|    YCrCb     |       0       |    07.54 sec     |  94.94 %    |
|    YCrCb     |       1       |    05.29 sec     |  96.71 %    |
|    YCrCb     |       2       |    09.71 sec     |  94.23 %    |
|    YCrCb     |      ALL      |    27.86 sec     |  99.22 %    |
|||||

Depending upon the training time and accuracy of the classifier, following are the selected parameters:

		colorspace = 'YUV'
		hog_channel = "ALL"
		orient = 9
		pix_per_cell = 8
		cell_per_block = 2

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

- After importing 8792 vehicle and 8792 non-vehicle images, I have used both color and gradient features from those images for training classifier. 
- Gradient are related to the shape of the objects while colors features are related to the color of the object in an image.
- I have used both the features which gives a better result than using an individual one.
- Code for extracting gradient features is in **code cell 4th**. Function: `get_hog_features()`
- Code for extracting color features is in **code cell 6th**. Function: `color_hist()`
- Parameters for feature extraction are selected in **code cell 11**
- All the cars and non-cars features are extracted in **code cell 12**. Function used: `extract_features()`
- Data preparation is done in **code cell 13th and 14th**. Cars and non-cars features are standardized using `StandardScaler()`.
- Corresponding labels' array is created in code cell 14th.
- Using sklearn's `train_test_split()` function, data is data is randomized to avoid over-fitting and is split in training(80%) and testing(20%) in **code cell 15th**.
- In **code cell 17th** Linear Support Vector Classifier(SVC) is trained based on the prepared dataset.
- Accuracy of the trained classifier is tested in **code cell 18th**
- Prediction of the test-set data is done in **code cell 19th**.


---

### III. Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

- Code for this part is in `Sliding Window Search and Video.ipynb` IPython Notebook.
- Instead of blindly searching for cars in the image, I have used sliding window search technique.
- For cars at a distance from our vehicle, I have used smaller window size(96) in small region of interest as shown below  
![alt text][image3]

- For cars at not to far,  I have used comparatively bigger window size(128) in the region of interest  
![alt text][image4]

- For cars relatively closer, I have used larger window size(256) in the region of interest  
![alt text][image5]

- Feature extraction is done in all these windows, (pixel-by-pixel) and they are compared with the features of the cars using our trained classifier.
![alt text][image7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

- Ultimately I searched on two scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

- Car features found in first layer:
![alt text][image8]
- Car features found in second layer:
![alt text][image9]
- Car features found in third layer:
![alt text][image10]

- Car features found in all layer:
![alt text][image11]

---

### IV. Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]
![alt text][image17]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image18]
![alt text][image19]
![alt text][image20]
![alt text][image21]
![alt text][image22]
![alt text][image23]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image24]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Approach I took:**  

1. Deciding windows of various sizes in which car can be present.
2. Searching for features of the car in these windows.
3. Detecting the possible car boxes.
4. Applying heatmap technique to eliminate the false positives.
5. Creating labels from the heatmap image
6. Drawing boxes on the car detected portion of the image.

**Issues faced:**  

1. Adjusting the proper window size for the car detection.
2. False positives: When the heat threshold was increased to eliminate false positive, some frames with car also started not detecting the car.
3. In the bright color patch of the road, detection of white car is difficult.
4. Pipeline tends to fail under shadows, on bright roads, etc.

**Points to make pipeline Robust:**

1. Training of large amount of diverse data.
2. Stabilizing the fluctuation in the detected boxes.
3. Setting appropriate sizes of the searching windows.
4. Use running average method for `hot_windows` for better detection.