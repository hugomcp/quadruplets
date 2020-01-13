################################################################################
# STEP 1
################################################################################
# Create a dataset file in '.csv' format.
# Each row should follow the format:
#	filename, class_1, class_2, ... learning_validation_test_set_flag
# with learning_validation_test_set_flag: 0 (learning), 1 (validation), 2(test)
# 
# Example:
#	Abdullah_Gul_0001.jpg,21,0,4,4,1
#
# 	Input image: Abdullah_Gul_0001.jpg
#	Class labels: 21, 0, 4 and 4
#	learning_validation_test_set_flag: 1
################################################################################



################################################################################
# STEP 2
################################################################################
# scr_quadruplet_embeddings.py parameters


# ap.add_argument('-d', '--dataset', required=True, help='CSV dataset file')
# This is the path of the dataset file, created in STEP 1 	


# ap.add_argument('-i', '--input_folder', required=True, help='Data input folder')
# This is the path to the dataset source images

ap.add_argument('-o', '--output_folder', required=True, help='Results/debug output folder')
# This is the path of the output file, where the model will be saved, and the results (plots and scores) 
# stored

ap.add_argument('-b', '--batch_size', type=int, default=64, help='Learning batch size')
# Learning batch size

ap.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='Learning rate')
# Learning rate 

ap.add_argument('-e', '--epochs', type=int, default=100, help='Tot. epochs')
# Maximum number of epochs

ap.add_argument('-p', '--patience', type=int, default=0, help='Tot. epochs without improvement to stop')
# Maximum number of consecutive epochs without improvements in the validation step to early stop

ap.add_argument('-s', '--image_size', type=int, default=64, help='Image size')
# Image size used in preprocess time

ap.add_argument('-m', '--manifold_dimension', type=int, default=128, help='Manifold output dimension')
# Dimension of the embedding space
 
ap.add_argument('-a', '--alpha', type=float, default=1.0, help='Alpha penalty')
# Weight of the 

ap.add_argument('-f', '--features', default='1', help='Input features used to define targets')
# List of the features actually used in the learning step

ap.add_argument('-n', '--features_manifold', default='0', help='Input features manifold flags')
# Flags specifying when the different labels of a feature define a manifold
# 1=yes, 0=no
# i.e., dist(label "1", label "2") > dist(label "1", label "3") ?
# Example:
# For 4 input features: "0, 0, 1, 0" means that only the labels of the third feature should be condidered 
# as lying in a manifold. For any other label, only "equal/different" should be considered  
 
################################################################################


