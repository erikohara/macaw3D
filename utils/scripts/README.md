# Scripts folder

This folder contains the scripts used on a cluster.

## 3D folder

### Cropping the images

The file `crop.py` was used to generate a image file of each slice of each subject of the dataset.
To run the python script, we used the `crop.slurm` file. You can pass the slice as an argument after the command. For instance, if you want the slice 112, you would use the command `sbatch crop.slurm 112`. 
A better way to run all slices would be using array tasks.  

### Reshaping the slices per slice

It took a long time to concatenate and reshape all the images, and we used that step to encode, to train the model, and to generate the images. Hence, we run a separate script `data_reshape.py` to concatenate, reshape the images, and save the output as a .npy file.  
To run the python script, we used the `data_reshape.slurm` file. You can pass the slice as an argument after the command. For instance, if you want the slice 105, you would use the command `sbatch data_reshape.slurm 105`. 
A better way to run all slices would be using array tasks.  

### Encoding the images per slice

The file `PCA-slice.py` was used to encode all the images of each slice .
To run the python script, we used the `PCA-slice.slurm` file. You can pass the slice as an argument after the command. For instance, if you want the slice 112, you would use the command `sbatch PCA-slice.slurm 112`. 
A better way to run all slices would be using array tasks.  

### Training the Model with PCA encoded images per slice

The file `Training-UKBB-MRI-Age-PCA.py` was used to train the model with all the PCA endoded images of each slice.
To run the python script, we used the `Training-UKBB-MRI-Age-PCA.slurm` file. You can pass the slice as an argument after the command. For instance, if you want the slice 112, you would use the command `sbatch Training-UKBB-MRI-Age-PCA.slurm 112`. 
A better way to run all slices would be using array tasks.  

### Generating Counterfactual Images per slice

The file `cf-generation.py` was used to generate counterfactual images on the PCA slice per slice model.
To run the python script, we used the `cf-generation.slurm` file. You can pass the initial and final slice as arguments after the command. For instance, if you want the images from slice 50 to 149, you would use the command `sbatch crop.slurm 50 149`. 
