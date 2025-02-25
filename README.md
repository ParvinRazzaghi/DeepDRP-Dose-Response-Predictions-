DeepDRP: Dose-Response Predictions of Drug Pairs Using Deep Learning Based on Data-Driven Feature Representation and Dose-Response Curve Characteristics 

To run the code, follow these steps:
1.	Download the required GitHub project (https://github.com/tkipf/keras-gcn.git) and Unzip the downloaded project and place its contents in the same directory as your project files. Then, complete the installation process. Move the extracted project files to the root folder of your project directory
2.	Download the following GitHub project (https://github.com/aalto-ics-kepaco/comboFM.git) and Move the extracted project files to the root folder of your project directory.
3.	Please set the epochs to a proper value in the code. The epochs number should be set in lines ? and ?. 
4.	Run the code. 

Note :During import TFFMRegressor (line 10), if you get the following error, please open python3.11/dist-packages/tffm/core.py  and then in line 96 replace “tf.train.AdamOptimizer” with “tf.optimizers.Adam”.
AttributeError: module 'tensorflow._api.v2.train' has no attribute 'AdamOptimizer'
