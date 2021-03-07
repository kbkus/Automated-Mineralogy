# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from skimage import io, color
import time
from sklearn.cluster import KMeans
import imutils
from min_dict import MINERALS

def sample_img(df, ppl, xpl, labels, width = 200, height = 200, inc = 75, status = False):
    '''
    Function samples image and stores RGB and greyscale color channels in dataframe.
    Assumed ppl and xpl images are the same size.
    ''' 
    width = width
    height = height
    inc = inc
    
    # convert images to Image type
    if type(ppl) == np.ndarray:
        ppl = Image.fromarray(ppl)
        
    if type(xpl) == np.ndarray:
        xpl = Image.fromarray(xpl)
        
    if type(labels) == np.ndarray:
        labels = Image.fromarray(labels)

    topleft = [0,0]
    botright = [width,height]
    # increment for x and y to change topleft and botright values
    xinc = inc
    yinc = inc
    # number of x slices to take and number of y slices to take
    xi = (ppl.size[0]-width)//inc
    yi = (ppl.size[1]-height)//inc
    
    # dictionary to hold orientations and their pairs
    orientations = {'leftright': Image.FLIP_LEFT_RIGHT,
                   'topbottom': Image.FLIP_TOP_BOTTOM}
    
    print(f'{yi} rows and {xi} columns to slice.')
    print(f'Total observations: {yi*xi*12}')
    
    # iterate through each sliding window along the height of the image
    start = time.time()
    for y in np.arange(yi):
        for x in np.arange(xi):
            
            # select orientation of image
            for orientation in [0, 1, 2]:
                if orientation == 0:
                    # isolate a crop of the original images
                    ppl_slice = ppl.crop(topleft+botright)
                    xpl_slice = xpl.crop(topleft+botright)
                    labels_slice = labels.crop(topleft+botright)
                elif orientation == 1:
                    # isolate a crop of the original images
                    ppl_slice = ppl.crop(topleft+botright).transpose(Image.FLIP_LEFT_RIGHT)
                    xpl_slice = xpl.crop(topleft+botright).transpose(Image.FLIP_LEFT_RIGHT)
                    labels_slice = labels.crop(topleft+botright).transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    # isolate a crop of the original images
                    ppl_slice = ppl.crop(topleft+botright).transpose(Image.FLIP_TOP_BOTTOM)
                    xpl_slice = xpl.crop(topleft+botright).transpose(Image.FLIP_TOP_BOTTOM)
                    labels_slice = labels.crop(topleft+botright).transpose(Image.FLIP_TOP_BOTTOM)

                # for each slice, rotate 90 degrees up to 270 to get 4 orientations of each slice
                for rotation in [0, 90, 180, 270]:
                    ppl_slice = ppl_slice.rotate(rotation)
                    xpl_slice = xpl_slice.rotate(rotation)
                    labels_slice = labels_slice.rotate(rotation)
                    # split RGB color channels
                    ppl_red, ppl_green, ppl_blue = ppl_slice.split()
                    xpl_red, xpl_green, xpl_blue = xpl_slice.split()
                    # put all colors into dataframe row
                    df_slice = pd.DataFrame([[np.array(ppl_red), np.array(ppl_green), np.array(ppl_blue), 
                                              np.array(xpl_red), np.array(xpl_green), np.array(xpl_blue),
                                              np.array(labels_slice), rotation, topleft]], columns = columns)
                    #append to main dataframe
                    df = df.append(df_slice, ignore_index = True)
            
            # shift sliding window
            topleft[0] += 100
            botright[0] += 100
            
        if status == True and y % 10 == 0:
            end = time.time()
            print(f'Runtime for row {y}/{yi} is {round(end - start,2)} seconds for {len(df)} samples')
            
        # reset sample box to left side of image
        topleft[0] = 0
        botright[0] = width
        # shift sample box down 100 pixels
        topleft[1] += 100
        botright[1] += 100
        
    
    return df

def show_img(df, rows):
    '''
    Function to display a slice from the dataframe. Rows is either a single value,
    or a list of integers you want to get the samples at those rows for.
    '''
    for i in rows:
        
        ppl = np.dstack((df['r_ppl'][i], df['g_ppl'][i], df['b_ppl'][i])).astype(np.uint8)
        xpl = np.dstack((df['r_xpl'][i], df['g_xpl'][i], df['b_xpl'][i])).astype(np.uint8)
        labels = df['labels'][i]
        
        plt.figure(figsize=(10,8))
        plt.subplot(2,3,1)
        plt.imshow(ppl)
        plt.subplot(2,3,2)
        plt.imshow(xpl)
        plt.subplot(2,3,3)
        plt.imshow(labels)

        plt.subplot(2,2,3)
        io.imshow(color.label2rgb(labels, ppl))
        plt.subplot(2,2,4)
        io.imshow(color.label2rgb(labels, xpl))
        
        
    return

def img_quant(img, n_colors=7, plot=False, min_dict=MINERALS):
    '''Transforms RGB image into a discretized array by clustering similar RGB values'''
    if min_dict == None:
        original = np.array(img)

        # reshape original into one long array with RGB values in lists
        X = original.reshape(-1,3)
        # fit the KMeans clusters
        kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(X)

        #assign each pixel to its proper cluster color based on the k-means labels
        segmented_img = kmeans.cluster_centers_[kmeans.labels_]

        # get a list of the unique colors (classes)
        # and re-label the image with discretized values
        # axis=0 so it will consider each list as a unique value
        unique_classes, relabeled_img = np.unique(segmented_img, axis=0, return_inverse=True)

        # re-shape to original image dimensions
        segmented_img = segmented_img.reshape(original.shape).astype(np.uint8)
        relabeled_img = relabeled_img.reshape((original.shape[0:2])).astype(np.uint8)
        relabeled_img = Image.fromarray(relabeled_img)
    
    '''Use this if you want to change colors based on set pixel values
    instead of clustering'''
    
    if min_dict != None:
        original = np.array(img)
        length, width = original.shape[0:2]
        original = original.reshape(-1,3)

        segmented_img = original.copy()

        # empty list to keep track of which minerals are present in the image
        unique_classes = []

        for mineral in min_dict.keys():  
            # make all pixels of one color a new color
            try:
                segmented_img[(original[:,0]==min_dict[mineral][0][0]) & (original[:,1]==min_dict[mineral][0][1]) & (original[:,2]==min_dict[mineral][0][2])] = min_dict[mineral][1]
                unique_classes.append(min_dict[mineral][1])
            except:
                print(f'{mineral} may not be present')

            # assign final to be only one column of values
                continue

        segmented_img = [x if x in unique_classes else 0 for x in segmented_img[:,0]]
        segmented_img = np.array(segmented_img).reshape(length, width).astype(np.uint8)
        relabeled_img = Image.fromarray(segmented_img)

        original = original.reshape(length,width,3)
    
    # plot if true
    if plot == True:
        plt.figure(figsize=(10,20))
        plt.subplot(1,2,1)
        plt.imshow(original)
        plt.title('Original image')
        
        plt.subplot(1,2,2)
        plt.imshow(segmented_img)
        plt.title(f'{len(unique_classes)} color clusters')
        
    return relabeled_img

def align_images(df, ppl_img, xpl_img, labels_img, threshold=10, confidence=0.99):
    '''Takes in a dataframe of corresponding points from each image and aligns them using
    a homography matrix. Images just need to be opened with Image.open()'''

    # get pts from df
    ppl_pts = np.array(df['ppl'].tolist())
    xpl_pts = np.array(df['xpl'].tolist())
    labels_pts = np.array(df['labels'].tolist())

    # compute homography matrix between two sets of matched points
    (xpl_H, xpl_mask) = cv2.findHomography(xpl_pts, ppl_pts, cv2.RANSAC, ransacReprojThreshold=threshold, confidence=confidence)
    (labels_H, labels_mask) = cv2.findHomography(labels_pts, ppl_pts, cv2.RANSAC, ransacReprojThreshold=threshold, confidence=confidence)
    
    # use the homography matrix to align the images to the ppl img
    (xpl_h, xpl_w) = ppl_img.shape[:2]
    xpl_aligned = cv2.warpPerspective(xpl_img, xpl_H, (xpl_w,xpl_h))
    
    (labels_h, labels_w) = ppl_img.shape[:2]
    labels_aligned = cv2.warpPerspective(labels_img, labels_H, (labels_w,labels_h))
    
    ppl_aligned = ppl_img

    # resize both the aligned and template images so we can easily 
    # visualize them on the screen
#     ppl_aligned = imutils.resize(ppl_img, width=700)
#     xpl_aligned = imutils.resize(xpl_aligned, width=700)
#     labels_aligned = imutils.resize(labels_aligned, width=700)

#     ppl = imutils.resize(ppl_img, width=700)
#     xpl = imutils.resize(xpl_img, width=700)
#     labels = imutils.resize(labels_img, width=700)

    ppl = ppl_img
    xpl = xpl_img
    labels = labels_img

    # side-by-side comparison of the output aligned image and the template
    stacked = np.hstack([ppl_aligned, xpl_aligned, labels_aligned])

    # second image alignment visualization will be overlaying the
    # aligned image on the template to get an idea of how good
    # the image alignment is

    template = ppl_aligned.copy()
    xpl_overlay = xpl_aligned.copy()
    labels_overlay = labels_aligned.copy()

    cv2.addWeighted(template, 0.5, xpl_overlay, 0.5, 0, xpl_overlay)
    cv2.addWeighted(template, 0.5, labels_overlay, 0.5, 0, labels_overlay)

    cv2.addWeighted(template, 0.5, xpl, 0.5, 0, xpl)
    cv2.addWeighted(template, 0.5, labels, 0.5, 0, labels)
    
    # stack overlay imgs next to each other
    overlays_stacked = np.hstack([xpl, labels])
    aligned_overlays_stacked = np.hstack([xpl_overlay, labels_overlay])

    # show the two output inmage alignment visualizations
    plt.figure(figsize=(30,20))

    plt.subplot(3,1,1)
    plt.title('Side-by-side images')
    plt.imshow(stacked)
    
    plt.subplot(3,1,2)
    plt.title('Un-aligned overlayed images')
    plt.imshow(overlays_stacked)

    plt.subplot(3,1,3)
    plt.title('Aligned overlayed images')
    plt.imshow(aligned_overlays_stacked)

    plt.tight_layout()
    
    return ppl_aligned, xpl_aligned, labels_aligned

def find_dimensions(df):
    
    '''Function to find the [height and width of labeled image'''
    
    width = df['labels'][1][0]-df['labels'][0][0]
    height = df['labels'][1][1]-df['labels'][0][1]
            
    return height, width

def resize_imgs(df, ppl, xpl, labels, method=Image.NEAREST, plot=False):
    
    ''' Load images as cv2.imread(). This will then find the matching crop
    dimensions using find_dimensions and return the new resized and cropped images'''
    
    # check if input images are np.ndarray type or Image type
    if type(ppl) == np.ndarray:
        ppl = Image.fromarray(ppl)
        xpl = Image.fromarray(xpl)
        labels = Image.fromarray(labels)


    # crop images
    ppl_crop = ppl.crop(df['ppl'][0]+df['ppl'][1])
    xpl_crop = xpl.crop(df['xpl'][0]+df['xpl'][1])
    labels_crop = labels.crop(df['labels'][0]+df['labels'][1])

    # get height and width that you need to resize all images to
    height, width = find_dimensions(df)
    # resize images
    # do not resize labels because we want to preserve pixel resolution
    ppl_crop = ppl_crop.resize((width, height),method)
    xpl_crop = xpl_crop.resize((width, height),method)
    
    if plot == True:
        plt.figure(figsize=(15, 20))
        plt.subplot(2,3,1)
        plt.title('Original PPL')
        plt.imshow(ppl)
        plt.subplot(2,3,2)
        plt.title('Original XPL')
        plt.imshow(xpl)
        plt.subplot(2,3,3)
        plt.title('Original Labels')
        plt.imshow(labels)
        # second row cropped images
        plt.subplot(2,3,4)
        plt.title('Cropped PPL')
        plt.imshow(ppl_crop)
        plt.subplot(2,3,5)
        plt.title('Cropped XPL')
        plt.imshow(xpl_crop)
        plt.subplot(2,3,6)
        plt.title('Cropped Labels')
        plt.imshow(labels_crop)
        plt.tight_layout()
    return ppl_crop, xpl_crop, labels_crop