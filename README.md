# czf
we propose a deep learning framework, named MDADDI, to predict potential DDIs. It consists of a molecular feature module,a DDI graph feature module,and an adaptive fusionmodule.
![Fig 1改](https://github.com/user-attachments/assets/67d24adc-a257-44cf-96b3-df72c9a3a305)
# Requirements
+ python >= 3.7
+ torch >= 1.12.0

+ torchvision >= 1.12.0+cu113
+ RDkit >= 2019.03.30
+ numpy >= 1.19.4
+ pandas >= 1.3.4

# Using
1.create_predata.py: Process the data
2.ours.py: the construction of the neural network
3.training.py: start file for model training
