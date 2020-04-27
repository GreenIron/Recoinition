#-*- coding: utf-8 -*-
"""
Created on Sat Mar 05 13:55:06 2016
@author: victor
"""

import os
from PIL import Image
from Algorithm import Piece

# File path
sample_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Coin_Samples', 'F.534_2_1899_2 (copy).jpg')
# Show Image
Image.open(sample_path).show()
# Import Coin (Piece in French)
toto = Piece.Piece(sample_path)
# Generate Obverse (Avers in French) and Reverse (Revers in French)
toto.generate_aversrever()
# Show circles coins
toto.show_coin_with_circles()
# Uncoil the coin and generate a stripe for both obverse and reverse
toto.avers.coin2stripe()
toto.revers.coin2stripe()
# Show stripes for obverse and reverse
toto.avers.show_stripe()
toto.revers.show_stripe()
# Extract words
toto.avers.extract_words()

# Get letters
#toto.avers.stripe2letters()



