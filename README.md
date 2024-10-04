# DNFSA-DDI
we propose a deep learning framework, named DNFSA-DDI, to predict potential DDIs. It consists of a molecular feature module,a DDI graph feature module,and an adaptive fusionmodule.
![Uploading a2b172e468374776dfec1e079c7a57a.jpg…]()

# Requirements
+ python = 3.7
+ torch = 1.12.0
+ torchvision = 1.12.0+cu113
+ RDkit = 2019.03.30
+ numpy = 1.19.4
+ pandas = 1.3.4

# Using
1. create_predata.py: Process the data
2. ours.py: the construction of the neural network  
3. training.py: start file for model training
