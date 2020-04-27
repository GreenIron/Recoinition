# -*- coding: utf-8 -*-  http://www.8bitavenue.com/2013/10/python-character-image-generator/

# View random image
#import os
#import numpy as npos.path.realpath(__file__)
#from random import randint
#from PIL import Image
#img_dim = 48
#train_dir = '/home/victor/Travail/Python_Projects/coin_OCR/training_set'
#source_images='coinletter_images.npy'
#train_data = np.load(os.path.join(train_dir,source_images))
#for n in range(0,10):
    #image = np.reshape(train_data[randint(0, 30000),:,:], [img_dim, img_dim]).astype('uint8')
    #Image.fromarray(image).show()

# Image.fromarray(image).save(r'/home/victor/Travail/Python_Projects/coin_OCR/Pieces temoins/SlicedLetters/1.jpeg')

#-------------------------------- Imports ------------------------------#

# Import operating system lib
import os, re, sys
from shutil import copyfile

# Import python imaging libs
from PIL import Image, ImageDraw, ImageFont
from skimage.filters import gaussian, sobel
from scipy.ndimage.interpolation import rotate
from skimage import feature, exposure
from skimage.util import random_noise
from scipy.signal import convolve2d
from skimage.draw import line_aa
import numpy as np
from numpy.random import rand

sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)),'django_coinsite_project/') )
sys.path.insert(0, os.path.join(os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)),'django_coinsite_project/Algorithm/') )
print(os.path.join(os.path.abspath(os.path.join(os.path.join(os.path.realpath(__file__), os.pardir), os.pardir)),'django_coinsite_project/'))
from Algorithm.Preprocessing import Preprocessing

# Import random generator
from random import randint

#-------------------------------- Cleanup ------------------------------#

def Cleanup():
    # Delete ds_store file
    if os.path.isfile(font_dir + '.DS_Store'):
        os.unlink(font_dir + '.DS_Store')

    # Delete all files from output directory
    for file in os.listdir(out_dir):
        file_path = os.path.join(out_dir, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
            
    # Delete all files from font directory
    for file in os.listdir(font_dir):
        file_path = os.path.join(font_dir, file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
            
    return

#--------------------------- Extract a selection of fonts. -----------------------#
def ExtractFont():
    for dirname, dirnames, filenames in os.walk(os.path.join(font_dir,'fonts')):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            # List and extract Bold and Medium fonts files in directory
            fontnames = [filename for filename in os.listdir(os.path.join(os.path.join(font_dir,'fonts'), subdirname)) if (re.search('-Bold.ttf',filename) or re.search('-Medium.ttf',filename))]
            for fontname in fontnames:
                copyfile(os.path.join(dirname, os.path.join(subdirname, fontname)),
                         os.path.join(os.path.join(font_dir,'selected_fonts'), fontname))

#--------------------------- Generate Characters -----------------------#
def GenerateCoinLetters():
    
    # Number of characters to be generated : nb_repeatloop * fontnames * characters * font_sizes * background_colors * light_types * light_levels * sigma_blurs
    nb_images = nb_repeatloop * len(os.listdir(os.path.join(font_dir,'selected_fonts'))) * len(characters) * len(font_sizes) * len(background_colors) * len(light_types) * len(noise_ratios) * len(light_levels) * len(sigma_blurs)
    print('GenerateCoinLetters will generate %i images of labelled coin letters' % nb_images)
            
    # Initialize the outputs
    coinletter_images = np.nan*np.zeros((nb_images,image_crop_size,image_crop_size,1))
    coinletter_labels = np.nan*np.zeros((nb_images,len(characters)))
    coinletter_labels_indexes = np.nan*np.zeros((nb_images))
    
    # Image counter
    k = 0
    
    # Process the font files
    for n_repeat in np.arange(nb_repeatloop):# Repeat the global process
        for fontnames in os.listdir(os.path.join(font_dir,'selected_fonts')),:
            # For each font do
            for fontname in fontnames:
                # Get font full file path
                font_resource_file = os.path.join(os.path.join(font_dir,'selected_fonts'), fontname)
                print(fontname)
                for n_char, char in enumerate(characters):# For each character do
                    for font_size in font_sizes: # For each font size do
                        for background_color in background_colors: # For each background color do
                            for light_type in light_types:
                                for noise_ratio in noise_ratios:
                                    for light_level in light_levels: # Contrast
                                        for sigma_blur in sigma_blurs:
                                            # Convert the character into unicode
                                            character = char
                    
                                            # Create character image : 
                                            # Grayscale, image size, background color
                                            char_image = Image.new('L', (image_size, image_size), 0)
                    
                                            # Draw character image
                                            imgTrain = ImageDraw.Draw(char_image)
                    
                                            # Specify font : Resource file, font size
                                            font = ImageFont.truetype(font_resource_file, font_size)
                    
                                            # Get character width and height
                                            (font_width, font_height) = font.getsize(character)
                                            margin_width = character_spacing*font_width # margin of the other characters around the center character
                                            character_before = characters[ int(len(characters)*np.random.uniform()) ]
                                            (font_width_before, _) = font.getsize(character_before)
                                            character_after  = characters[ int(len(characters)*np.random.uniform()) ]
                                                                                    
                                            #pour detecter lettre faire projection de edge
                                                                                
                                            # Set x position
                                            x = (image_size - font_width )/2 - font_width_before
                                            # Set y position. Include some randomness
                                            y = (image_size - (1.15+rnd_ypos*np.random.rand())*font_height)/2
                    
                                            # Draw text : Position, String. Add two random letters before and after
                                            # Options = Fill color, Font
                                            imgTrain.text((x-margin_width, y), character_before, 1, font=font)
                                            imgTrain.text((x+font_width_before, y), character, 1, font=font)
                                            imgTrain.text((x+font_width_before+font_width+margin_width, y), character_after, 1, font=font)
                                            
                                            char_image_array = np.asarray(char_image)
                                            
                                            # Speckle noise => salt and pepper => blur to mimick wear. Will have to replicate the preprocessing done in the final app.
                                            char_image_array_filtered = (char_image_array + noise_ratio*np.random.randn(image_size,image_size))/2.0
                                            
                                            char_image_array_filtered = char_image_array_filtered/np.max(np.max(char_image_array_filtered))
                                            char_image_array_filtered = random_noise(char_image_array_filtered, mode='salt', amount=0.02)
                                            
                                            char_image_array_filtered = gaussian(char_image_array_filtered, sigma=sigma_blur)
                                            
                                            # Random rotation
                                            rot_angle = max_rot_angle*(np.random.rand()-0.5)*2
                                            char_image_array_filtered = rotate(char_image_array_filtered, rot_angle, reshape=False)
                                                                                    
                                            # Gradient/Sobel mask to mimick natural lights (sobel or sobel h or morphological gradient)
                                            if light_type==1:
                                                # Light from left to right
                                                diff_kernel = np.array([[ -1, +1]])/2.0
                                                #diff_kernel = np.transpose(np.array([[ +1, +2, +1],[ +0, +0, +0],[ -1, -2, -1]])/4.0)
                                                char_image_edge = convolve2d(char_image_array_filtered, diff_kernel, boundary='symm', mode='same') # or feature.canny(char_image_array)*1.0
                                            elif light_type==2:
                                                # Light from right to left
                                                diff_kernel = np.array([[ +1, -1]])/2.0
                                                char_image_edge = convolve2d(char_image_array_filtered, diff_kernel, boundary='symm', mode='same') # or feature.canny(char_image_array)*1.0
                                            elif light_type==3:
                                                # Light top right to bottom
                                                diff_kernel = np.transpose(np.array([[ -1, +1]])/2.0)
                                                char_image_edge = convolve2d(char_image_array_filtered, diff_kernel, boundary='symm', mode='same') # or feature.canny(char_image_array)*1.0
                                            elif light_type==4:
                                                # Light bottom right to top
                                                diff_kernel = np.transpose(np.array([[ +1, -1]])/2.0)
                                                char_image_edge = convolve2d(char_image_array_filtered, diff_kernel, boundary='symm', mode='same') # or feature.canny(char_image_array)*1.0
                                            elif light_type==5:
                                                # light from above (edge detection)
                                                char_image_edge = 1.0-sobel(char_image_array_filtered)
                                            
                                            # Coin type generation of the letter to create the relief
                                            
                                            # Normalize the image
                                            char_image_array_filtered = 255.0*(char_image_edge-np.min(np.min(char_image_edge)))/(np.max(np.max(char_image_edge))-np.min(np.min(char_image_edge)))
                                            # Shift the image
                                            char_image_array_filtered = char_image_array_filtered + light_level
                                            
                                            char_image_array_filtered[char_image_array_filtered<0] = 0
                                            char_image_array_filtered[char_image_array_filtered>255] = 255
                                            
                                            # Inverse random rotation with a little bit of noise : rnd_rot/2 deg
                                            char_image_array_filtered = rotate(char_image_array_filtered, -rot_angle + rnd_rot*(np.random.rand()-0.5), reshape=False)
                                            
                                            # Crop the Image
                                            char_image_array_filtered = char_image_array_filtered[(crop_margin+1):-crop_margin,(crop_margin+1):-crop_margin]
                                            
                                            # Add lines. If the center character is empty, then we add stipes
                                            if True:#char==' ':
                                                #char_image_array_filtered = np.zeros((image_crop_size,image_crop_size))
                                                rr, cc, val = line_aa( int(image_crop_size*rand()), 0, int(image_crop_size*rand()), image_crop_size-1) # int(image_crop_size*rand())
                                                char_image_array_filtered[rr, cc] = np.float64(val * 255)                                            
                                            
                                            # Preprocessing
                                            char_image_array_filtered = Preprocessing.contrast_stretching(Preprocessing.median_filtering(char_image_array_filtered.astype('uint8'))).astype('uint8')
                                                                                                                                                                                
                                            # Thresholding
                                            #char_image_array_filtered = Preprocessing.thresholding(char_image_array_filtered, window_ratio=0.3, k=0.6).astype('uint8')
                                                                                        
                                            Image.fromarray(char_image_array_filtered).show()
                                            
                                            """
                                            # Preprocessing to be consistent with the input images. Notably with thresholding
                                            char_image_array_filtered = Preprocessing.Preprocessing.threshold(char_image_array_filtered ,'Niblack')
                                            Image.fromarray(char_image_array_filtered).show()
                                            
                                            char_image_array_filtered = Preprocessing.Preprocessing.threshold(char_image_array_filtered ,'Sauvola')
                                            Image.fromarray(char_image_array_filtered).show()
                                            """
                                                                                                    
                                                                        
                                            # Update the dataset
                                            # mnist.train.images is a tensor (an n-dimensional array) with a shape of [55000, 784].
                                            # The first dimension is an index into the list of images and the second dimension is the index for each pixel in each image.
                                            # Each entry in the tensor is a pixel intensity between 0 and 1, for a particular pixel in a particular image.
                                            # Dimensions are (num_images, rows, cols, 1)
                                            coinletter_images[k,:,:,0] = char_image_array_filtered.astype('uint8') # np.reshape(char_image_array_filtered,np.prod(np.shape(char_image_array_filtered)),1)
                                            # mnist.train.labels is a [55000, 10] array of floats
        
                                            coinletter_labels[k,:] = np.zeros(len(characters))
                                            coinletter_labels[k,n_char] = 1.0
                                            coinletter_labels_indexes[k] = n_char
                                            
                                            # Save image
                                            k = k + 1
    
    return coinletter_images, coinletter_labels, coinletter_labels_indexes

#------------------------------- Input and Output ------------------------#
# https://fonts.google.com/?category=Serif&subset=latin&thickness=6&slant=1&selection.family=Playfair+Display:700
# font selection criteria = latin, serif, not slant, medium thick
# Directory containing fonts
font_dir = os.path.join(os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)),'font_sources/') # Output
out_dir = os.path.join(os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir)),'training_set/')

#---------------------------------- Characters ---------------------------#

# Numbers
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# Capital letters
capital_letters = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Select characters
characters = numbers + capital_letters

#---------------------------------- Parameters -------------------------------#

"""
TIRER UN BATCH DE CETTE TAILLE, TAILLE AUSSI GRANDE QUŃ VEUT MAISSIMPLEMENT DIVISER LA TAILLE du fichier sauvegardé

ensuite effectivement iterer sur les fichiers sur lesquels on fait le training...

il faut avoir environ 100k de donneee

si ca marche pas essayer avec un reseau as hoc avec modules...

aire autant de parametres qu on veut mais tirer aleatoirement les parametres a chaque batch

du coup, on peut garder le script

juste specifier la taille qu on veut pour un batch d image et le nimbre de fois

taille batch

avoir une seule boucle for et ieterer

il faut ensuite iterer aussi sur maketrain

FAIRE CLASSE DEDIEE POUR GENERATION ET APPRENTISSAGE

ENSUITE LANCER CA DEPUIS UN SCRIPT
"""


# Background color
background_colors = (100,)

nb_repeatloop = 1
contrast = 10
noise_ratios = (0.01, 0.03, 0.15,0.25,0.28)
sigma_blurs = (1.2, 1.4) # 
light_levels = (0, 50) #,50,100 
max_rot_angle = 25

character_spacing = 0.20 # spacing between characters

rnd_rot = 25 # random rotation range

rnd_ypos = 0.4

# eventuellement  rajouter des traits ou droites

#----------------------------------- Sizes -------------------------------#

# Character sizes
font_sizes = (46, 50, 54, 58) # small_sizes + medium_sizes + large_sizes # 

# image_crop_size
image_crop_size = 4*12
image_size = int((image_crop_size - 1)/(1-2*(np.cos(np.pi/4)-np.cos(np.pi/4+max_rot_angle*np.pi/180.0))))
crop_margin = int((np.cos(np.pi/4)-np.cos(np.pi/4+max_rot_angle*np.pi/180.0))*image_size)
image_size = image_crop_size + 2*crop_margin + 1

# Light types
light_types = (1, 2, 3, 4, 5) # eclairage dans differentes directions

#----------------------------------- Main --------------------------------#
print('Start generate_trainingset')
# Do cleanup
print('Cleanup')
Cleanup()

# Install fonts
print('ExtractFont')
ExtractFont()

# Generate characters
print('GenerateCoinLetters')
coinletter_images, coinletter_labels, coinletter_labels_indexes = GenerateCoinLetters()
# Multi-dimensional arrays are only shuffled along the first axis:
shuffled_index = np.arange(len(coinletter_labels))
np.random.shuffle(shuffled_index)

coinletter_images         = np.float32(coinletter_images[shuffled_index,:,:,:])
coinletter_labels         = np.float32(coinletter_labels[shuffled_index,:])
coinletter_labels_indexes = np.float32(coinletter_labels_indexes[shuffled_index])

#Image.fromarray(coinletter_images[0,:,:,0]).show()
print(np.nonzero(coinletter_labels[0,:])) # 'np.nonzero(coinletter_labels[0,:]= %s' % 
print(np.shape(coinletter_images)) # 'np.shape(coinletter_images)= %s' % 
print(np.shape(coinletter_labels)) # 'np.shape(coinletter_labels)= %s' % 
print(np.shape(coinletter_labels_indexes)) # 'np.shape(coinletter_labels)= %s' % 

# Save this data set: original letter, labels and generic information
print('Saving images and labels')
np.save(os.path.join(out_dir,'coinletter_images'), coinletter_images)
np.save(os.path.join(out_dir,'coinletter_labels'), coinletter_labels)
np.save(os.path.join(out_dir,'coinletter_labels_indexes'), coinletter_labels_indexes)
with open(os.path.join(out_dir,'coinletter_readme'), "w") as text_file:
    text_file.write("GenerateCoinLetters configuration : contrast= %s, noise_ratio= %s, sigma_blurs= %s, light_levels= %s, max_rot_angle= %s, background_colors= %s"
                    % (contrast, noise_ratios, sigma_blurs, light_levels, max_rot_angle, background_colors) )
    
print('generate_trainingset is done')
