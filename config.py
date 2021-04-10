# configuration vars we want to set in one place

# imshape should have 3 channels for rgb input images
# (height, width)
imshape = (200, 200, 3)

# set classification mode (binary or multi)
mode = 'multi'

# model_name (unet or fcn_8)
model_name = 'unet_'+mode

# log dir for tensorboard
logbase = 'logs'

# classes are defined as 0 through 6 (7 total)
n_classes = 7

# number of epochs
n_epochs = 2

# Config variables for resize.py and tiepoints.py

# paths for original images
ppl_path = '/Users/kacikus/Dropbox/AutomatedMineralogy_Project/Automated-Mineralogy/Images/mn16-07-ppl.jpg'
xpl_path = '/Users/kacikus/Dropbox/AutomatedMineralogy_Project/Automated-Mineralogy/Images/mn16-07-cpl.jpg'
labels_path = '/Users/kacikus/Dropbox/AutomatedMineralogy_Project/Automated-Mineralogy/Images/MN16-07.png'

# path to save tiepoints and resize points pkl files
resize_pts_path = '/Users/kacikus/Dropbox/AutomatedMineralogy_Project/Data/mn1607_resize_pts.pkl'
tiepoints_path = '/Users/kacikus/Dropbox/AutomatedMineralogy_Project/Data/mn1607_tiepoints.pkl'