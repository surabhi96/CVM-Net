This project is a cross-view localization package based on CVM-Net. 

Find the necessary train and validation scripts in the CVM-Net folder. File details are mentioned in CVM-Net/Readme.md

Add the dataset in the Data folder and the pretrained models in the Model folder. 

generate_dataset folder contains files for generating data suitable for our pipeline. Details are mentioned in generate_dataset/readme.md

For convenience, a conda environment is provided (cvmnet_environment.yml) with packages required by CVM-Net. 
Run $ conda env create -f cvmnet_environmet.yml && conda activate cvm-net  
