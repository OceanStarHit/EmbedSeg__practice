# %%
from tqdm import tqdm
from glob import glob
import tifffile
import numpy as np
import os
from EmbedSeg.utils.preprocess_data import extract_data, split_train_val, split_train_crops, get_data_properties
from EmbedSeg.utils.generate_crops import *
import json

# %% [markdown]
# ### Download Data

# %%
data_dir = '../../../data'
project_name = 'Mouse-Skull-Nuclei-CBG'

# %% [markdown]
# Ideally, <b>*.tif</b>-type images and the corresponding masks should be respectively present under <b>images</b> and <b>masks</b>, under directories <b>train</b>, <b>val</b> and <b>test</b>, which can be present at any location on your workstation, pointed to by the variable <i>data_dir</i>. (In order to prepare such instance masks, one could use the Fiji plugin <b>Labkit</b> as detailed <b>[here](https://github.com/juglab/EmbedSeg/wiki/Use-Labkit-to-prepare-instance-masks)</b>). The following would be the desired structure as to how data should be present. 
# 
# <img src="../../../static/png/01_dir_structure.png" width="100"/>
# 
# If you already have your data available in the above style, please skip to the <b><a href="#center">third</a></b> section of this notebook, where you specify the kind of center to which constitutive pixels of an object should embed. 
# Since for the <b> Mouse-Skull-Nuclei-CBG</b> dataset, we do not have the data in this format yet, we firstly download the data from an external url in the following cells, next we split this data to create our `train`, `val` and `test` directories. 

# %% [markdown]
# The images and corresponding masks are downloaded from an external url, specified by `zip_url` to the path specified by the variables `data_dir` and `project_name`. The following structure is generated after executing the `extract_data`, `split_train_test` and `split_train_val` methods below:
# 
# <img src="../../../static/png/07_mouse-skull-nuclei-cbg.png" width="500"/>

# %%
extract_data(
    zip_url = 'https://github.com/juglab/EmbedSeg/releases/download/v0.1.0/Mouse-Skull-Nuclei-CBG.zip',
    data_dir = data_dir,
    project_name = project_name,
)

# %% [markdown]
# ### Split Data into `train`, `val` \& `test`

# %% [markdown]
# Now, we would like to reserve a small fraction (15 % by default) of the available train dataset as validation data. Here, in case you would like to repeat multiple experiments with the same partition, you may continue and press <kbd>Shift</kbd> + <kbd>Enter</kbd> on the next cell - but in case, you would like different partitions each time, please add the `seed` attribute equal to a different integer (For example, 
# ```
# split_train_val(
# data_dir = data_dir, 
# project_name = project_name, 
# train_val_name = 'train', 
# subset = 0.15,
# seed = 1000)
# ```
# )

# %%
split_train_val(
    data_dir = data_dir,
    project_name = project_name, 
    train_val_name = 'train',
    subset = 0.0)

# %% [markdown]
# ### Calculate some dataset specific properties 

# %% [markdown]
# In the next cell, we will calculate properties of the data such as `min_object_size`, `foreground_weight` etc. <br>
# We will also specify some properties, for example,  
# * set `data_properties_dir['one_hot'] = True` in case the instances are encoded in a one-hot style. 
# * set `data_properties_dir['data_type']='16-bit'` if the images are of datatype `unsigned 16 bit` and 
#     `data_properties_dir['data_type']='8-bit'` if the images are of datatype `unsigned 8 bit`.
# 
# Lastly, we will save the dictionary `data_properties_dir` in a json file, which we will access in the `02-train` and `03-predict` notebooks.

# %%
one_hot = False
data_properties_dir = get_data_properties(data_dir, project_name, train_val_name=['train'], 
                                          test_name=['test'], mode='3d', one_hot=one_hot)

data_properties_dir['data_type']='16-bit'
data_properties_dir['pixel_size_x_microns']=0.073 # set equal to voxel size (microns) in x dimension
data_properties_dir['pixel_size_y_microns']=0.073 # set equal to voxel size (microns) in y dimension
data_properties_dir['pixel_size_z_microns']=0.20 # set equal to voxel size (microns) in z dimension

with open('data_properties.json', 'w') as outfile:
    json.dump(data_properties_dir, outfile)
    print("Dataset properies of the `{}` dataset is saved to `data_properties.json`".format(project_name))

# %% [markdown]
# ### Specify desired centre location for spatial embedding of pixels

# %% [markdown]
# Interior pixels of an object instance can either be embedded at the `centroid` (evaluated in $\mathcal{O(n)}$ operations, where $\mathcal{n}$ is the number of pixels in an object instance), or the `medoid` (evaluated in $\mathcal{O(n^{2})}$ operations). Please note that evaluating `medoid` of the instances could be slow especially if you choose a large `crop_size` later: in such a scenario, a quicker alternative is opting for a higher <b><a href='#speed_up'>`speed_up`</a></b> factor.

# %%
center = 'medoid' # 'medoid', 'centroid'
try:
    assert center in {'medoid', 'centroid'}
    print("Spatial Embedding Location chosen as : {}".format(center))
except AssertionError as e:
    e.args += ('Please specify center as one of : {"medoid", "centroid"}', 42)
    raise



# %% [markdown]
# ### Specify cropping configuration parameters

# %% [markdown]
# Images and the corresponding masks are cropped into patches centred around an object instance, which are pre-saved prior to initiating the training. Here, `data_subsets` is a list of names of directories which is processed. <br>
# Note that the cropped images, masks and center-images would be saved at the path specified by `crops_dir` (The parameter `crops_dir` is set to ```./crops``` by default, which creates a directory at the same location as this notebook). The `anisotropy_factor` is set equal to the ratio of voxel sizes in z to voxel sizes in x or y. <br>
# In case, there are out-of-memory issues or cropping takes too long, please try increasing the <b>`speed_up`</b> parameter by steps of 1. 

# %% [markdown]
# <a id="speed_up"></a>

# %%
crops_dir = './crops'
data_subsets = ['train'] 
crop_size_z = 96 
crop_size_x = 128 
crop_size_y = 128 
anisotropy_factor = data_properties_dir['pixel_size_z_microns']/data_properties_dir['pixel_size_x_microns']
speed_up = 3

# %% [markdown]
# ### Generate Crops

# %% [markdown]
# <div class="alert alert-block alert-warning"> 
#     The cropped images and masks are saved at the same-location as the example notebooks. <br>
#     Generating the crops would take a little while!
# </div>

# %%
for data_subset in data_subsets:
    image_dir = os.path.join(data_dir, project_name, data_subset, 'images')
    instance_dir = os.path.join(data_dir, project_name, data_subset, 'masks')
    image_names = sorted(glob(os.path.join(image_dir, '*.tif'))) 
    instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))  
    for i in tqdm(np.arange(len(image_names))):
        process_3d(image_names[i], instance_names[i], os.path.join(crops_dir, project_name), data_subset, 
                crop_size_x = crop_size_x, crop_size_y = crop_size_y, crop_size_z = crop_size_z,
                center = center, anisotropy_factor = anisotropy_factor, speed_up = speed_up)
    print("Cropping of images, instances and centre_images for data_subset = `{}` done!".format(data_subset))

# %%
split_train_crops(project_name = project_name, center = center, crops_dir = crops_dir, subset = 0.15)

# %%



