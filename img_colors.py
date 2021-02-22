def show_color_counts(img):
    ''' Input an image in the format of a np.array and return
    subplots of each pixel color plotted with the RGB color value and number
    of pixels in the image that are that color.'''
    img = img.reshape((-1, 3))
    values, counts = np.unique(img, axis=0, return_counts=True)
    
plt.imshow([[[0, 155, 85]]])