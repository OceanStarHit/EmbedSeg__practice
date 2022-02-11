# %%
import numpy as np
import os
from EmbedSeg.train import begin_training
from EmbedSeg.utils.create_dicts import create_dataset_dict, create_model_dict, create_loss_dict, create_configs
import torch
from matplotlib.colors import ListedColormap
import json
# comment the following line, if running in the headless mode
# %matplotlib tk  

# %% [markdown]
# ### Specify the path to `train`, `val` crops and the type of `center` embedding which we would like to train the network for:

# %% [markdown]
# The train-val images, masks and center-images will be accessed from the path specified by `data_dir` and `project-name`.
# <a id='center'></a>

# %%
data_dir = 'crops'
project_name = 'Mouse-Skull-Nuclei-CBG'
center = 'medoid' # 'centroid', 'medoid'

print("Project Name chosen as : {}. \nTrain-Val images-masks-center-images will be accessed from : {}".format(project_name, data_dir))

# %%
try:
    assert center in {'medoid', 'centroid'}
    print("Spatial Embedding Location chosen as : {}".format(center))
except AssertionError as e:
    e.args += ('Please specify center as one of : {"medoid", "centroid"}', 42)
    raise

# %% [markdown]
# ### Obtain properties of the dataset 

# %% [markdown]
# Here, we read the `dataset.json` file prepared in the `01-data` notebook previously.

# %%
if os.path.isfile('data_properties.json'): 
    with open('data_properties.json') as json_file:
        data = json.load(json_file)
        one_hot, data_type, foreground_weight, n_z, n_y, n_x, pixel_size_z_microns, pixel_size_x_microns = data['one_hot'], data['data_type'], float(data['foreground_weight']), int(data['n_z']), int(data['n_y']), int(data['n_x']), float(data['pixel_size_z_microns']), float(data['pixel_size_x_microns'])

# %%
normalization_factor = 65535 if data_type=='16-bit' else 255

# %% [markdown]
# ### Specify training dataset-related parameters

# %% [markdown]
# Some hints: 
# * The `train_size` attribute indicates the number of image-mask paired examples which the network would see in one complete epoch. Ideally this should be the number of `train` image crops. For the `Mouse-Skull-Nuclei-CBG` dataset, we obtain ~ 128 crops, since this is a small number, we set `train_size` to double the size 256. 
# * The effective batch size is determined as a product of the attributes `train_batch_size` and `virtual_train_batch_multiplier`. For example, one could set a small `batch_size` say equal to 2 (to fit in one's GPU memory), and a large `virtual_train_batch_multiplier` say equal to 8, to get an effective batch size equal to 16. 
# 
# In the cell after this one, a `train_dataset_dict` dictionary is generated from the parameters specified here!

# %%
train_size = 256
train_batch_size = 2 
virtual_train_batch_multiplier = 8 

# %% [markdown]
# ### Create the `train_dataset_dict` dictionary  

# %%
train_dataset_dict = create_dataset_dict(data_dir = data_dir, 
                                         project_name = project_name,  
                                         center = center, 
                                         size = train_size, 
                                         batch_size = train_batch_size, 
                                         virtual_batch_multiplier = virtual_train_batch_multiplier, 
                                         normalization_factor= normalization_factor,
                                         type = 'train',
                                         name = '3d')

# %% [markdown]
# ### Specify validation dataset-related parameters

# %% [markdown]
# Some hints:
# * The size attribute indicates the number of image-mask paired examples which the network would see in one complete epoch. Here, it is recommended to set `val_size` equal to the total number of validation image crops. For example, for the `Mouse-Skull-NucleiCBG` dataset, we notice ~22 validation crops, since this is a small number, hence we set `val_size = 176`.
# * The effective batch size is determined as a product of the attributes `val_batch_size` and `virtual_val_batch_multiplier`. Here at times, it is okay to set a higher effective batch size for the validation dataset than the train dataset, since evaluating on validation data consumes lesser GPU memory.
# 
# In the cell after this one, a `val_dataset_dict` dictionary is generated from the parameters specified here!

# %%
val_size = 176
val_batch_size = 16
virtual_val_batch_multiplier = 1

# %% [markdown]
# ### Create the `val_dataset_dict` dictionary

# %%
val_dataset_dict = create_dataset_dict(data_dir = data_dir, 
                                       project_name = project_name, 
                                       center = center, 
                                       size = val_size, 
                                       batch_size = val_batch_size, 
                                       virtual_batch_multiplier = virtual_val_batch_multiplier,
                                       normalization_factor= normalization_factor,
                                       type ='val',
                                       name ='3d')

# %% [markdown]
# ### Specify model-related parameters

# %% [markdown]
# Some hints:
# * Set the `input_channels` attribute equal to the number of channels in the input images. 
# * Set the `num_classes = [6, 1]` for `3d` training and `num_classes = [4, 1]` for `2d` training
# <br>(here, 6 implies the offsets and bandwidths in x, y and z dimensions and 1 implies the `seediness` value per pixel)
# 
# In the cell after this one, a `model_dataset_dict` dictionary is generated from the parameters specified here!

# %%
input_channels = 1
num_classes = [6, 1] 

# %% [markdown]
# ### Create the `model_dict` dictionary

# %%
model_dict = create_model_dict(input_channels = input_channels,
                              num_classes = num_classes,
                              name = '3d')

# %% [markdown]
# ### Create the `loss_dict` dictionary

# %%
loss_dict = create_loss_dict(n_sigma = 3, foreground_weight = foreground_weight)

# %% [markdown]
# ### Specify additional parameters 

# %% [markdown]
# Some hints:
# * The `n_epochs` attribute determines how long the training should proceed. In general for reasonable results, you should atleast train for longer than 50 epochs.
# * The `display` attribute, if set to True, allows you to see the network predictions as the training proceeds. 
# * The `display_embedding` attribute, if set to True, allows you to see some sample embedding as the training proceeds. Setting this to False leads to faster training times.
# * The `save_dir` attribute identifies the location where the checkpoints and loss curve details are saved. 
# * If one wishes to **resume training** from a previous checkpoint, they could point `resume_path` attribute appropriately. For example, one could set `resume_path = './experiment/Mouse-Organoid-Cells-CBG-demo/checkpoint.pth'` to resume training from the last checkpoint.
# 

# %%
n_epochs = 200
display = True
display_embedding = False
save_dir = os.path.join('experiment', project_name+'-'+'demo')
resume_path  = save_dir+'/checkpoint.pth'
display_zslice = 48

# %% [markdown]
# In the cell after this one, a `configs` dictionary is generated from the parameters specified here!
# <a id='resume'></a>

# %% [markdown]
# ### Create the  `configs` dictionary 

# %%
configs = create_configs(n_epochs = n_epochs,
                         display = display, 
                         display_embedding = display_embedding,
                         resume_path = resume_path, 
                         save_dir = save_dir, 
                         n_z = n_z,
                         n_y = n_y, 
                         n_x = n_x,
                         anisotropy_factor = pixel_size_z_microns/pixel_size_x_microns,
                         display_zslice = display_zslice)

# %% [markdown]
# ### Begin training!

# %% [markdown]
# Executing the next cell would begin the training. 
# 
# If `display` attribute was set to `True` above, then you would see the network predictions at every $n^{th}$ step (equals 5, by default) on training and validation images. 
# 
# Going clockwise from top-left is 
# 
#     * the raw-image which needs to be segmented, 
#     * the corresponding ground truth instance mask, 
#     * the network predicted instance mask, and 
#     * (if display_embedding = True) from each object instance, 5 pixels are randomly selected (indicated with `+`), their embeddings are plotted (indicated with `.`) and the predicted margin for that object is visualized as an axis-aligned ellipse centred on the ground-truth - center (indicated with `x`)  for that object
# 

# %%
begin_training(train_dataset_dict, val_dataset_dict, model_dict, loss_dict, configs)

# %% [markdown]
# <div class="alert alert-block alert-warning"> 
#   Common causes for errors during training, may include : <br>
#     1. Not having <b>center images</b> for  <b>both</b> train and val directories  <br>
#     2. <b>Mismatch</b> between type of center-images saved in <b>01-data.ipynb</b> and the type of center chosen in this notebook (see the <b><a href="#center"> center</a></b> parameter in the third code cell in this notebook)   <br>
#     3. In case of resuming training from a previous checkpoint, please ensure that the model weights are read from the correct directory, using the <b><a href="#resume"> resume_path</a></b> parameter. Additionally, please ensure that the <b>save_dir</b> parameter for saving the model weights points to a relevant directory. 
# </div>

# %%



