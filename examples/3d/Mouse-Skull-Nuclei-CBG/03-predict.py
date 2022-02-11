# %%
import itk
import itkwidgets
from itkwidgets import view
from ipywidgets.embed import embed_minimal_html
import tifffile
import numpy as np

# %%
from EmbedSeg.utils.create_dicts import create_test_configs_dict
from EmbedSeg.test import begin_evaluating
from glob import glob
import tifffile
import matplotlib.pyplot as plt
from EmbedSeg.utils.visualize import visualize
import os
from matplotlib.colors import ListedColormap
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import json

# %% [markdown]
# ### Specify the path to the evaluation images

# %%
data_dir = '../../../data'
project_name = 'Mouse-Skull-Nuclei-CBG'
print("Evaluation images shall be read from: {}".format(os.path.join(data_dir, project_name)))

# %% [markdown]
# ### Specify evaluation parameters 

# %% [markdown]
# Some hints:
# * `tta`: Setting this to True (default) would enable **test-time augmentation**
# * `ap_val`: This parameter ("average precision value") comes into action if ground truth segmentations exist for evaluation images, and allows to compare how good our predictions are versus the available ground truth segmentations.
# * `seed_thresh`: This parameter ("seediness threshold") allows considering only those pixels as potential instance-centres which have a seediness score greater than `seed_thresh`
# * `checkpoint_path`: This parameter provides the path to the trained model weights which you would like to use for evaluation. One could test the pretrained model (available at `'../../../pretrained_models/Mouse-Skull-Nuclei-CBG/best_iou_model.pth'`) to get a quick glimpse on the results.
# * `save_dir`: This parameter specifies the path to the prediction instances. Equal to `static` by default.
# * `save_images`: If True, this saves predictions at `./static/predictions/` 
# * `save_results`: If True, this saves results at `./static/results/`
# 
# In the cell after this one, a `test_configs` dictionary is generated from the parameters specified here!
# <a id='checkpoint'></a>

# %%
# uncomment for the model trained by you
# checkpoint_path = os.path.join('experiment', project_name+'-'+'demo', 'best_iou_model.pth')
# if os.path.isfile('data_properties.json'): 
#     with open('data_properties.json') as json_file:
#         data = json.load(json_file)
#         one_hot = data['one_hot']
#         data_type = data['data_type']
#         min_object_size = int(data['min_object_size'])
#         foreground_weight = float(data['foreground_weight'])
#         n_z, n_y, n_x = int(data['n_z']),int(data['n_y']), int(data['n_x'])
#         pixel_size_z_microns, pixel_size_y_microns, pixel_size_x_microns = float(data['pixel_size_z_microns']), float(data['pixel_size_y_microns']), float(data['pixel_size_x_microns']) 
#         avg_background_intensity = float(data['avg_background_intensity'])

#use the following for the pretrained model weights
checkpoint_path = os.path.join('../../../pretrained_models', project_name, 'best_iou_model.pth')
if os.path.isfile(os.path.join('../../../pretrained_models', project_name,'data_properties.json')): 
    with open(os.path.join('../../../pretrained_models', project_name, 'data_properties.json')) as json_file:
        data = json.load(json_file)
        one_hot = data['one_hot']
        data_type = data['data_type']
        min_object_size = int(data['min_object_size'])
        foreground_weight = float(data['foreground_weight'])
        n_z, n_y, n_x = int(data['n_z']),int(data['n_y']), int(data['n_x'])
        pixel_size_z_microns, pixel_size_y_microns, pixel_size_x_microns = float(data['pixel_size_z_microns']), float(data['pixel_size_y_microns']), float(data['pixel_size_x_microns']) 
        avg_background_intensity = float(data['avg_background_intensity'])

# %% [markdown]
# ℹ️ Setting `tta=True` would give better results but would take longer to compute!

# %%
tta = True
ap_val = 0.5
seed_thresh = 0.90
save_dir = './static'
save_images = True
save_results = True
normalization_factor = 65535 if data_type=='16-bit' else 255

# %%
if os.path.exists(checkpoint_path):
    print("Trained model weights found at : {}".format(checkpoint_path))
else:
    print("Trained model weights were not found at the specified location!")

# %% [markdown]
# ### Create `test_configs` dictionary from the above-specified parameters

# %%
test_configs = create_test_configs_dict(data_dir = os.path.join(data_dir, project_name),
                                        checkpoint_path = checkpoint_path,
                                        tta = tta, 
                                        ap_val = ap_val,
                                        seed_thresh = seed_thresh, 
                                        min_object_size = min_object_size, 
                                        save_images = save_images,
                                        save_results = save_results,
                                        save_dir = save_dir,
                                        normalization_factor = normalization_factor,
                                        one_hot = one_hot,
                                        n_z = n_z,
                                        n_y = n_y,
                                        n_x = n_x,
                                        anisotropy_factor = pixel_size_z_microns/pixel_size_x_microns,
                                        name = '3d',
                                        )

# %% [markdown]
# ### Begin Evaluating

# %% [markdown]
# Setting `verbose` to True shows you Average Precision at IOU threshold specified by `ap_val` above for each individual image. The higher this score is, the better the network has learnt to perform instance segmentation on these unseen images.

# %%
begin_evaluating(test_configs, verbose = False, avg_bg = avg_background_intensity/normalization_factor)

# %% [markdown]
# <div class="alert alert-block alert-warning"> 
#   Common causes for a low score/error is: <br>
#     1. Accessing the model weights at the wrong location. Simply editing the <b> checkpoint_path</b> would fix the issue. <br>
#     2. At times, you would notice an improved performance by lowering <b><a href="#checkpoint"> seed_thresh</a></b> from 0.90 (default) to say 0.80. <br>
#     3. CUDA error: out of memory - ensure that you shutdown <i>02-train.ipynb</i> notebook before running this notebook.
# </div>

# %% [markdown]
# ### Visualize some predictions

# %% [markdown]
# Here, we use the `itkwidgets` to first display any one of the evaluation images and then display the corresponding prediction by the model. Please feel free to change the `index` to look at other predictions.

# %%
if(save_images):
    prediction_file_names = sorted(glob(os.path.join(save_dir,'predictions','*.tif')))
    ground_truth_file_names = sorted(glob(os.path.join(save_dir,'ground-truth','*.tif')))
    image_file_names = sorted(glob(os.path.join(save_dir, 'images','*.tif')))

# %%
index = 0
print("Image filename is {} and index is {}".format(os.path.basename(image_file_names[index]), index))

image = normalization_factor*tifffile.imread(image_file_names[index])
prediction = tifffile.imread(prediction_file_names[index])

image_itk =itk.GetImageFromArray(image)
image_itk.SetSpacing([pixel_size_x_microns, pixel_size_y_microns, pixel_size_z_microns])
prediction_itk =itk.GetImageFromArray(prediction)
prediction_itk.SetSpacing([pixel_size_x_microns, pixel_size_y_microns, pixel_size_z_microns])
view(image_itk, label_image=prediction_itk, cmap=itkwidgets.cm.BrBG, annotations=False, vmax=800, ui_collapsed=True, background=(192, 192, 192))
#embed_minimal_html('export_'+str(index)+'.html', views=viewer, title='Widgets export')

# %%



