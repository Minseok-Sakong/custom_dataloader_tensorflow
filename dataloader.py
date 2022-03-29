import numpy as np
import nibabel as nib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import Sequence
import sklearn
import random
import matplotlib.pyplot as plt

PATCH_SIZE = 64
num_classes = 33
pixels = PATCH_SIZE*PATCH_SIZE*PATCH_SIZE
channel_nums = 3

def pre_func_example(input):
	m = np.mean(input)
	s = np.std(input)
	new_image = (input - m) / s
	return new_image

class DataSet_random(Sequence):
	"""
	brain_filenames = [list] a list containing absolute paths of image files
	mask_filenames = [list] a list contating absolute paths of mask files
	batch_size = [int] batch size
	shuffle = [boolean] if true, shuffle images and masks on each epoch
	pre_func = [function] a function to normalize/standardize input files
	"""

	def __init__(self, brain_filenames, mask_filenames, batch_size, shuffle=False, pre_func=None):
		self.brain_filenames = brain_filenames
		self.mask_filenames = mask_filenames
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.pre_func = pre_func
		if self.shuffle:
			self.on_epoch_end()
		assert len(self.brain_filenames)==len(self.mask_filenames), "Images and masks do not match."

	def __len__(self):
		return int(len(self.mask_filenames))

	def __getitem__(self, index):
		'''
		returns randomly cropped patches(= # of batch size) from a subject image file
		'''
		brain_name_batch = self.brain_filenames[index]
		if self.mask_filenames is not None:
			mask_name_batch = self.mask_filenames[index]

		# channel_nums depends on the input image. Ex) RGB image: channel_nums=3, Grayscale image: channel_nums=1
		brain_batch = np.zeros((self.batch_size, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, channel_nums), dtype='float32')
		mask_batch = np.zeros((self.batch_size, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, num_classes), dtype='float32')

		# load nifti files to numpy arrays
		image = np.array(nib.load(brain_name_batch).get_fdata())
		mask = np.array(nib.load(mask_name_batch).get_fdata())

		# standardize/normalize image intensity value
		if self.pre_func is not None:
			new_image = self.pre_func(image)

		#Input image file is a 3D image
		x = new_image.shape[0]
		y = new_image.shape[1]
		z = new_image.shape[2]
		cnt = 0

		while cnt < self.batch_size:
			rand_x = random.randint(0, x - PATCH_SIZE - 1)
			rand_y = random.randint(0, y - PATCH_SIZE - 1)
			rand_z = random.randint(0, z - PATCH_SIZE - 1)

			new_brain = np.zeros(shape=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
			new_brain = new_image[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE, rand_z:rand_z + PATCH_SIZE]
			new_mask = np.zeros(shape=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
			new_mask = mask[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE, rand_z:rand_z + PATCH_SIZE]
			# Background pixel value = 0
			background_pixels = np.count_nonzero(new_mask == 0)

			background_percentage = background_pixels / pixels
			# Check background is over 90% in a patch. If true, discard it.
			if (background_percentage > 0.90):
				continue
			else:
				train_brain = np.stack((new_brain,) * channel_nums, axis=-1)
				train_mask = np.expand_dims(new_mask, axis=3)
				train_mask_cat = to_categorical(train_mask, num_classes=num_classes)

				brain_batch[cnt] = train_brain
				mask_batch[cnt] = train_mask_cat
				cnt += 1

		return brain_batch, mask_batch

	def on_epoch_end(self):
		if (self.shuffle):
			self.brain_filenames, self.mask_filenames = sklearn.utils.shuffle(self.brain_filenames, self.mask_filenames)
		else:
			pass
