# **Finding Lane Lines on the Road** 

## Michael DeFilippo

### This project was used to introduce the concepts of using OpenCV and the Python programming language to find lane lines on a road. The general idea is to first identify lane lines on a set of images using various computer vision techniques and then apply your lane finding pipeline to actual video of a car driving down a road.  

---

**Finding Lane Lines on the Road**

The pipeline that I designed uses the following techniques to detect lane lines:
  * Color Filtering
  * Canny Edge Detection
  * Region of Interest Selection
  * Hough Transforms and Line Detection
  
### Lane lines were first detected using these techniques on the following test images
![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]




[//]: # (Image References)

[image10]: ./examples/grayscale.jpg "Grayscale"
[image1]: ./test_images/solidWhiteCurve.jpg  
[image2]: ./test_images/solidYellowCurve2.jpg  
[image3]: ./test_images/solidYellowLeft.jpg
[image4]: ./test_images/solidWhiteRight.jpg  
[image5]: ./test_images/solidYellowCurve.jpg   
[image6]: ./test_images/whiteCarLaneSwitch.jpg

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, I converted the images to grayscale, then I .... 

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by ...

If you'd like to include images to show how the pipeline works, here is how to include an image: 




### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
