#####################################
#     @author Hamidreza Abooei      #
#####################################

# import Libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

dir = "./Data/data1/"
image_name1 = "1 ("
image_name2 = ").jpg"

image_number = 1

location = dir + image_name1 + str(image_number) + image_name2 
# import image
img = cv.imread(location, 0)
# get shape of image v: colomn | s: row 
s, v= img.shape
circle_numbers = 0
#Threshold 
th = 360
flag = True
while((circle_numbers != 1) | (flag)):
    # Hough transform
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, s//10,
                            param1=th, param2=20, minRadius=s//20, maxRadius=s//5)

    circle_numbers = np.size(circles) / 3
    final_circle = 0
    if (circle_numbers >= 1):

        th = th + 1
        circle_numbers = 0
        for circle in circles[0, :]:
            if ((circle[0]>0.2*s) & (circle[0]<0.5*s) & (circle[1]>0.4 * v) & (circle[1]<0.8 * v) ):
                circle_numbers += 1
                final_circle = circle
        if (circle_numbers==1):
            flag =False
        if (circle_numbers == 0):
            th = th - 2
    else:
        th = th - 1

# convert to integer
circles = np.uint16(np.around(final_circle))

#initial point
init = (circles[1],circles[0])

# Sharpening the image 
sharpen = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
img_sharped = cv.filter2D(img,ddepth=-1 , kernel = sharpen)


def region_growing(img , seed , threshold):
    #initialize seed
    x,y = seed
    # print(x,y)
    #define a dark image inorder to Mask
    black = np.zeros(np.shape(img),dtype = np.uint8)
    #define clicked image white:
    black[x,y] = 1
    #this variable will difine whether we finished hole filling 
    end_finder = 0
    #start loop for define whole hole section
    while (np.sum(black) > end_finder):
        # count the number of points that are detected yet
        end_finder = np.sum(black)
        # define a cross for dilate kernel
        kernel = cv.getStructuringElement(cv.MORPH_CROSS,(3,3))
        # dilate section with + kernel
        black_dilated = cv.dilate(black,kernel)
        new_pixels_black = (black_dilated - black )
        new_pixels = new_pixels_black * img
        base_intensity = np.sum ( black * img ) / np.sum ( black )
        ret,img_thresh = cv.threshold(new_pixels,base_intensity - threshold,1,cv.THRESH_BINARY)
        black = cv.bitwise_or(black,img_thresh)

    return black


growthed = 0
region_thresh = 50


while((np.sum(growthed) < 0.001 * s * v )|( np.sum(growthed) > 0.2 * s * v )):
    # use region growing algorithm to find femur 
    growthed = region_growing(img,init , region_thresh)
    if(np.sum(growthed) < 0.01 * s * v ):
        region_thresh += 5
    else:
        region_thresh -= 5



# Apply closing to image
kernel =  np.ones((11,11), np.uint8)
final_mask = cv.morphologyEx(growthed,cv.MORPH_CLOSE,kernel)

# Apply mask to original image
segmented_image = final_mask * img


#apply one level convex_hull 
def convex_hull (img):
    shape = img.shape
    img_paded = np.pad (img , 1,mode = 'constant' , constant_values = (0 , 0))
    # Apply first kernel
    kernel1 = np.array((
        [1, 0,0],
        [1, -1,0],
        [1, 0,0]), dtype="int")
    out1 = cv.morphologyEx(img_paded, cv.MORPH_HITMISS, kernel1)
    out1 = np.bitwise_or(out1,img_paded)
    # Apply second kernel
    kernel = np.array((
        [1, 1, 1],
        [0,-1,0],
        [0,0,0]), dtype="int")
    out2 = cv.morphologyEx(out1, cv.MORPH_HITMISS, kernel)
    out2 = cv.bitwise_or(out2,out1)
    # Apply third kernel
    kernel = np.array((
        [0,0, 1],
        [0, -1, 1],
        [0,0, 1]), dtype="int")
    out3 = cv.morphologyEx(out2, cv.MORPH_HITMISS, kernel)
    out3 = cv.bitwise_or(out3,out2)
    # Apply fourth kernel
    kernel = np.array((
        [0,0,0],
        [0, -1, 0],
        [1, 1, 1]), dtype="int")
    out4 = cv.morphologyEx(out3, cv.MORPH_HITMISS, kernel)
    out4 = cv.bitwise_or(out4,out3)
    out4 = out4[1:shape[0]+1,1:shape[1]+1]
    return out4


# Apply 3 level convex hull
convexed_img = convex_hull(final_mask)
convexed_img = convex_hull(convexed_img)
convexed_img = convex_hull(convexed_img)

last_convexed_image = convexed_img * img


plt.figure()
plt.imshow(last_convexed_image ,cmap = "gray")
plt.imsave(dir +"res/"+ image_name1 + str(image_number) + image_name2 ,last_convexed_image,cmap = "gray")
plt.show()

