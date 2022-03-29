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



