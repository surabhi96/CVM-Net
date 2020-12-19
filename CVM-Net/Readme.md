The main file for training and checking accuracies is validate.py.
Note: Validate.py works with the sydney dataset in the current configuration. 

validate_nyc works with the nyc dataset on the fly. 

Validate script contains 
1) Train function
2) Loss function
3) Function to calculate accuracy 
4) TestImage class to find the top k retrieved satellite images (find_knn - See usage in test_server.py)
5) get_descriptors function to get descriptors of all the ground and satellite images in the train set
Note: The validate_nyc script contains comments on adding the data file paths, and pre-trained model paths along with tuning hyperparameters for training.  

The various input_data scripts modify the inputs for different experiments conducted. 
The default script used is input_data.py

train_google.py is the script used only for training with the Google sydney dataset. 
This functionality is now added inside validate script.
