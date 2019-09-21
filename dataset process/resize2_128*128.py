import os ,sys
import numpy as np
from PIL import Image
import random


ROOT_DIR = os.getcwd()
align_image_path = os.path.join(ROOT_DIR,'128*128')
origin_image = os.listdir( align_image_path )
oimage_amount = len(origin_image)
select_amt = oimage_amount

print('amount of image :' + str(select_amt))

select_file = [[0] for i in range(0,select_amt)]
for i in range(0,select_amt):
	select_file[i][0]= origin_image[i]  # random choose  numbers of pic
	img_file_name = './img_align_celeba/' + str(select_file[i][0])
	img = Image.open( img_file_name )
	
	img = img.resize((128, 128),Image.ANTIALIAS)
	img.save('./128*128/' + str(select_file[i][0]) )
	img.close()
	print('resize the ' + str(i) + ' th image')

