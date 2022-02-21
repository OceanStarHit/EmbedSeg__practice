# %%
import numpy as np
from EmbedSeg.utils.create_dicts import create_test_configs_dict
from EmbedSeg.test import begin_evaluating
from glob import glob
import tifffile
import matplotlib.pyplot as plt
from EmbedSeg.utils.visualize import visualize
from EmbedSeg.train import invert_one_hot
import os
from matplotlib.colors import ListedColormap
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import json

# %% [markdown]
# ### Specify the path to the evaluation images

# %%
data_dir = './data'
project_name = 'custom'
print("Evaluation images shall be read from: {}".format(os.path.join(data_dir, project_name)))

# %% [markdown]
# ### Specify evaluation parameters 

# %% [markdown]
# Some hints:
# * `tta`: Setting this to True (default) would enable **test-time augmentation**
# * `ap_val`: This parameter ("average precision value") comes into action if ground truth segmentations exist for evaluation images, and allows to compare how good our predictions are versus the available ground truth segmentations.
# * `seed_thresh`: This parameter ("seediness threshold") allows considering only those pixels as potential instance-centres which have a seediness score greater than `seed_thresh`
# * `checkpoint_path`: This parameter provides the path to the trained model weights which you would like to use for evaluation. One could test the pretrained model (available at `'../../../pretrained_models/dsb-2018/best_iou_model.pth'`) to get a quick glimpse on the results.
# * `save_dir`: This parameter specifies the path to the prediction instances. Equal to `static` by default.
# * `save_images`: If True, this saves predictions at `./static/predictions/` 
# * `save_results`: If True, this saves results at `./static/results/`
# 
# In the cell after this one, a `test_configs` dictionary is generated from the parameters specified here!
# <a id='checkpoint'></a>

# %%
# uncomment for the model trained by you
checkpoint_path = os.path.join('./examples/2d/custom','experiment', project_name+'-'+'demo', 'best_iou_model.pth')
if os.path.isfile('./examples/2d/custom/data_properties.json'): 
    with open('./examples/2d/custom/data_properties.json') as json_file:
        data = json.load(json_file)
        one_hot, data_type, min_object_size, n_y, n_x, avg_bg = data['one_hot'], data['data_type'], int(data['min_object_size']), int(data['n_y']), int(data['n_x']), float(data['avg_background_intensity'])
# use the following for the pretrained model weights
# checkpoint_path = os.path.join('../../../pretrained_models', project_name, 'best_iou_model.pth')
# if os.path.isfile(os.path.join('../../../pretrained_models', project_name,'data_properties.json')): 
#     with open(os.path.join('../../../pretrained_models', project_name, 'data_properties.json')) as json_file:
#         data = json.load(json_file)
#         one_hot, data_type, min_object_size, n_y, n_x, avg_bg = data['one_hot'], data['data_type'], int(data['min_object_size']), int(data['n_y']), int(data['n_x']), float(data['avg_background_intensity'])

# %%
tta = True
ap_val = 0.5
seed_thresh = 0.90
save_dir = './examples/2d/custom/static'
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
                                        # cuda=False,
                                        n_y = n_y,
                                        n_x = n_x,)

# %% [markdown]
# ### Begin Evaluating

# %% [markdown]
# Setting `verbose` to True shows you Average Precision at IOU threshold specified by `ap_val` above for each individual image. The higher this score is, the better the network has learnt to perform instance segmentation on these unseen images.

# %%
# %matplotlib agg
begin_evaluating(test_configs, verbose = False, avg_bg = avg_bg/normalization_factor)

# %% [markdown]
# <div class="alert alert-block alert-warning"> 
#   Common causes for a low score/error is: <br>
#     1. Accessing the model weights at the wrong location (this could happen, for example, if you access the prediction on a day different from the one when you trained the notebook). Simply editing the <b> checkpoint_path</b> would fix the issue. <br>
#     2. At times, you would notice an improved performance by lowering <b><a href="#checkpoint"> seed_thresh</a></b> from 0.90 (default) to say 0.80. <br>
#     3. CUDA error: out of memory - ensure that you shutdown <i>02-train.ipynb</i> notebook before running this notebook.
# </div>

# %% [markdown]
# <div class="alert alert-block alert-info"> 
# The complete set of runs for different partitions of the data is available <b><a href = "https://github.com/juglab/EmbedSeg/wiki/DSB_2018"> here </a></b>!
# </div>    

# %% [markdown]
# ### Load a glasbey-style color map

# %%
new_cmp= np.load('./cmaps/cmap_60.npy')
new_cmp = ListedColormap(new_cmp)

# %% [markdown]
# ### Investigate some qualitative results

# %% [markdown]
# Here you can investigate some quantitative predictions. GT segmentations and predictions, if they exist, are loaded from sub-directories under `save_dir`.
# Simply change `index` in the next two cells, to show the prediction for a random index.
# Going clockwise from top-left is 
# 
#     * the raw-image which needs to be segmented, 
#     * the corresponding ground truth instance mask, 
#     * the network predicted instance mask, and 
#     * (if display_embedding = True) from each object instance, 5 pixels are randomly selected (indicated with `+`), their embeddings are plotted (indicated with `.`) and the predicted margin for that object is visualized as an axis-aligned ellipse centred on the predicted - center (indicated with `x`)  for that object

# %%
# %matplotlib inline
if(save_images):
    prediction_file_names = sorted(glob(os.path.join(save_dir,'predictions','*.tif')))
    ground_truth_file_names = sorted(glob(os.path.join(save_dir,'ground-truth','*.tif')))
    embedding_file_names = sorted(glob(os.path.join(save_dir,'embedding','*.tif')))
    image_file_names = sorted(glob(os.path.join(data_dir, project_name, 'test', 'images','*.tif')))

# %%
if (save_images):
    index = 24
    print("Image filename is {} and index is {}".format(os.path.basename(image_file_names[index]), index))
    prediction = tifffile.imread(prediction_file_names[index])
    image = tifffile.imread(image_file_names[index])
    embedding = tifffile.imread(embedding_file_names[index])
    if len(ground_truth_file_names) > 0:
        ground_truth = tifffile.imread(ground_truth_file_names[index])
        visualize(image = image, prediction = prediction, ground_truth = ground_truth, embedding = embedding, new_cmp = new_cmp)
    else:
        visualize(image = image, prediction = prediction, ground_truth = None, embedding = embedding, new_cmp = new_cmp)

# %%
if (save_images):
    index = 26
    print("Image filename is {} and index is {}".format(os.path.basename(image_file_names[index]), index))
    prediction = tifffile.imread(prediction_file_names[index])
    image = tifffile.imread(image_file_names[index])
    embedding = tifffile.imread(embedding_file_names[index])
    if len(ground_truth_file_names) > 0:
        ground_truth = tifffile.imread(ground_truth_file_names[index])
        visualize(image = image, prediction = prediction, ground_truth = ground_truth, embedding = embedding, new_cmp = new_cmp)
    else:
        visualize(image = image, prediction = prediction, ground_truth = None, embedding = embedding, new_cmp = new_cmp)

# %%



