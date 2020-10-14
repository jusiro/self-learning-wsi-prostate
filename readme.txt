10/2020
Owner: Julio Silva-Rodríguez (jjsilva@upv.es)

CODE USAGE INSTRUCTIONS

1) Download a '.tiff' wsi dataset (in this work we used PANDA dataset) on '/data/slides/'. Create a ground truth and partition in an '.xlsx' file similar to '/data/partitions/partition_PANDA.xlsx'.

2) Perform patch extraction of this dataset usign the function '/code/patch_extraction.py'. Patches will be saved in 'data/patches/'.

3) Train teacher and student models using '/code/train_self_learning_gleason_grading.py'. Results will be stored in 'data/models/'.

4) For testing the models on external datasets, download the patches in other folder and create a ground truth data frame similar to '/data/partitions/gt_sicap_patches.xlsx'.

In our work, one of the external datasets validated was SICAPv2 (https://data.mendeley.com/datasets/9xxm58dvs3/1), which dataframe is shared in '/data/partitions/gt_sicap_patches.xlsx'.


REQUIRED PYTHON (3.6.7) LIBRARIES

os
cv2 4
PIL
matplotlib 3.0.3
random
timeit
pandas 0.25.2
numpy 1.16.2
datetime
torch 1.5.0
torchvision 0.6.0
scipy 1.4.1
openslide 1.1.1
