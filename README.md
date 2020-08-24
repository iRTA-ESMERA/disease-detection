# Disease Detection 

Component for determining the status of plant leaves (disease/health). The component takes as input one image dataset and returns the classification of each image as healthy/diseased.

## Dataset Used

The dataset we used was the PlantPathology Apple Dataset [1]. After downloading the dataset, we ***divided***  it into train/test folders and created the csv files manually. 
The existing **train.csv** contained all the names of all the files of the images. Specifically, 1821 images were named Test_(number_of_image).jpg and were used as the test dataset and 4371 images were used as the train dataset. We ***cut/pasted*** all the names of the test images to a new file called **test.csv**. 
Those csv files and the images were placed into a folder named **data**. Finally, we added the string ".jpg" to all the file names in the csv files (so as it is possible to be read with python libraries).
For example, inside the csv file, the name "Train_0" changed to "Train_0.jpg". The final folder structure is shown below.

## Architecture: Residual Neural Network

For the architecture of the neural network, residual blocks is used as in the original paper of He [2]. The only differences is that instead of the classic ReLU, leaky_ReLU is used and that in each convolution layers a l1_l2 kernel regularizer is used.

## Avoiding overfitting

In order to avoid overffiting we used both Early Stopping [3] and Learning Rate Reduction on Plateau [4].


## Folder Structure


disease-detection/

├── data/

│   ├── images_data/

│   │   ├── test/

│   │   │   └── Test_0.jpg

│   │   │   └── Test_1.jpg

│   │   │   └── 	...

│   │   └── train/

│   │   │   └── Train_0.jpg

│   │   │   └── Train_1.jpg

│   │   │   └──     ...

│   ├── test.csv

│   └── train.csv

└── disease_detection_AI_COMPONENT_NN.py


## References

[1] [PlantPathology Apple Dataset](https://www.kaggle.com/piantic/plantpathology-apple-dataset)

[2] [Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition, 2015](https://arxiv.org/abs/1512.03385)

[3] [Early Stopping](https://en.wikipedia.org/wiki/Early_stopping)

[4] [Learning Rate Reduction on Plateau](https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/)