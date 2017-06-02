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
![jpg](test_images/solidYellowCurve.jpg)

### Color Filtering
The images are uploaded with OpenCV in RGB (Red, Green, Blue) color space. From there apply a filter to only select the white threshold of the image. Next I convert the original image to HSL color space to more easily detect yellow colors using a yellow threshold. I then apply a filter to the HSL image to select the yellow threshold of the image. Finally I combine the images into one image with only white and yellow color selection.

```python
def color_filter(image):
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
```

![png](finished/solidYellowCurve_colorFilter_whiteFilter.png)

I apply a white thresholding and show that only white is visible in this image.

![png](finished/solidYellowCurve_colorFilter_hsl.png)

Here I have converted the original image to HSL color space

![png](finished/solidYellowCurve_colorFilter_yellowFilter.png)

I next apply yellow thresholding and show that only yellow is visible in this image. 

![png](finished/solidYellowCurve_colorFilter.png)

The white and yellow lines can be clearly identified in this photo. 


### Canny Edge Detection
Next I filtered the image using Canny edge detection techniques using OpenCV functions in the following order to the final image of the color filtering process. 
   * Grayscale
   * Gaussian Blur
   * Canny Edge Detection
   
```python
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
  
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
  
```
![png](finished/solidYellowCurve_grayscale.png)

First I converted the color filtered image to grayscale. 

![png](finished/solidYellowCurve_blurGrayscale.png)

Next I applied a Gaussian blur to the grayscale image with a kernel size of 5.

![png](finished/solidYellowCurve_canny.png)

I finished the process by applying a Canny edge detector to find the lines in the image. The goal is to find a setting of thresholds that detect enough edges in the image. I settled on a low threshold of 50 and a high threshold of 150 for the Canny edge detector function. The documentation in OpenCV references the selection of a threshold ratio between 3:1 or 2:1 (upper:lower).

### Region of Interest
I next
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
