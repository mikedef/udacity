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
## Pipeline

### Uploading Test Images  
Lane lines were first detected using these techniques on the provided test images. See image below for an example of the test images. 
![alt text][image2] 

### Color Filtering
The images are uploaded with OpenCV in RGB (Red, Green, Blue) color space. From there I apply a color filter to select only white and yellow colors of the image. 
def color_filter(image):
    '''
    Filter the image to only show white and yellow pixels
        *See Changing Colorspaces turorial on OpenCV
    ''' 
    ## Filter for the white pixles 
    # Define range of white color
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    
    # Threshold the image to get only the white colors
    white_mask = cv2.inRange(image, lower_white, upper_white)
    
    # Bitwise-AND mask and original image
    white_image = cv2.bitwise_and(image,image, mask= white_mask)
    
    ## Filter for the yellow pixles
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Convert image to HSL color space for enhanced color detection with shadows 
    hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
                       
    # Define range of yellow color
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    #Threshold the image to get only the yellow colors
    yellow_mask = cv2.inRange(hsl, lower_yellow, upper_yellow)
    
    # Bitwise-AND mask and original image
    yellow_image = cv2.bitwise_and(image,image, mask= yellow_mask)
    
    ## Combine (Blend) the images (see OpenCV-Arithmetic Operation on Images tutorial)
    combined_image = cv2.addWeighted(white_image,1.0, yellow_image,1.0, 0)
   
    return combined_image
    '''





[//]: # (Image References)

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
