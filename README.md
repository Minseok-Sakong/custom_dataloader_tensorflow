# custom_dataloader_tensorflow
Custom data loader for feeding randomly cropped patches to CNN model

## Example of usage
```
tr_ds = DataSet_random(brain_tr, mask_tr, 4,True, pre_func_example)
val_ds = DataSet_random(brain_val, mask_val, 4,False, pre_func_example)
history=model.fit(tr_ds,
            epochs=1000,
            verbose=1,
            validation_data=val_ds)
```

### Input parameters

+ brain_filenames = [list] a list containing absolute paths of image files
+ mask_filenames = [list] a list contating absolute paths of mask files
+ batch_size = [int] batch size
+ shuffle = [boolean] if true, shuffle images and masks on each epoch
+ pre_func = [function] a function to normalize/standardize input files

### Order of process
```
brain_name_batch = self.brain_filenames[index]
if self.mask_filenames is not None:
            mask_name_batch = self.mask_filenames[index]
```
+ Fetch an image/mask file from image/mask file list

```
brain_batch = np.zeros((self.batch_size, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, channel_nums), dtype='float32')
mask_batch = np.zeros((self.batch_size, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE, num_classes), dtype='float32')

# load nifti files to numpy arrays
image = np.array(nib.load(brain_name_batch).get_fdata())
mask = np.array(nib.load(mask_name_batch).get_fdata())
```
+ Initialize batch_size X an empty numpy array (=final return output that gets fed into the model)
+ Load image/mask file to numpy arrays

```
# standardize/normalize image intensity value
if self.pre_func is not None:
            new_image = self.pre_func(image)
```
+ Standardize or Normalize image intensity/pixel values

```
rand_x = random.randint(0, x - PATCH_SIZE - 1)
rand_y = random.randint(0, y - PATCH_SIZE - 1)
rand_z = random.randint(0, z - PATCH_SIZE - 1)

new_brain = np.zeros(shape=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
new_brain = new_image[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE, rand_z:rand_z + PATCH_SIZE]
new_mask = np.zeros(shape=(PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))
new_mask = mask[rand_x:rand_x + PATCH_SIZE, rand_y:rand_y + PATCH_SIZE, rand_z:rand_z + PATCH_SIZE]
```
+ Crop a random patch from the image (PATCH_SIZE x PATCH_SIZE x PATCH_SIZE)

```
if (background_percentage > 0.90):
            continue
else:
            train_brain = np.stack((new_brain,) * channel_nums, axis=-1)
            train_mask = np.expand_dims(new_mask, axis=3)
            train_mask_cat = to_categorical(train_mask, num_classes=num_classes)

            brain_batch[cnt] = train_brain
            mask_batch[cnt] = train_mask_cat
            cnt += 1
 ```
 + Check if background is over 90% in the cropped patch. If true, discard it and create another random patch
 + Due to the label imbalance problem(typical brain label mask has a label imbalance issue, where background is over 80% in the mask file), this helps the model to learn other labels well, besides the background label.
