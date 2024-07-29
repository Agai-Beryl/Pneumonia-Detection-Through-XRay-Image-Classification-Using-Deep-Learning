###  PAEDIATRIC PNEUMONIA DETECTION THROUGH XRAY IMAGE CLASSIFICATION :A DEEP LEARNING APPROACH

![pneumonia](https://github.com/user-attachments/assets/413c8c0a-d3e7-4b26-91b8-e6657a7fad21)

####  PROBLEM STATEMENT

In 2017, pneumonia was responsible for 15% of deaths in children under five years old, making it the most common cause of death in children.

Chest X-ray is the first line of investigation and commonly used in confirming infection and assessing the severity. However, the interpretation of chest X-rays can be tedious and requires experienced physicians. The chest X-ray findings may even vary among different interpreters.

Deep learning, particularly Convolutional Neural Networks(CNN), is an crucial asset in medical image analysis. By automatically extracting the important features, CNN uncovers hidden patterns in images and can accurately detect pneumonia in chest X-rays.

Training a CNN from scratch can be time-consuming and requires a vast image dataset for superior performance. However, a huge image dataset is always a challenge in the medical field. Alternatively, using pre-trained models such as VGG-16 can overcome this limitation, as these models have been pre-trained on a huge dataset. By applying transfer learning, these models can achieve superior performance even on a small dataset.

####  Main Objective

Our main objective is develop a deep learning-based model to accurately classify whether a pediatric patient has pneumonia from chest x-ray images,with at least 70% accuracy to improve diagnostic accuracy and efficiency in a clinical setting.

#### DATA UNDERSTANDING

The [dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download) is structured into three folders (train, test, val), each containing subfolders for the image categories (Pneumonia/Normal). It includes 5,863 X-ray images (JPEG) divided into two categories (Pneumonia/Normal). The chest X-ray images (anterior-posterior) were obtained from retrospective cohorts of pediatric patients aged one to five years from Guangzhou Women and Children’s Medical Center, Guangzhou. These images were taken as part of routine clinical care. For analysis, low-quality or unreadable scans were removed through an initial quality control screening. Two expert physicians graded the diagnoses of the images before they were used for training the AI system. To address any potential grading errors, a third expert also reviewed the evaluation set.

#### MODELLING
#### 1. CNN
The model has  6,203,681 total parameters where 6,202,785 are trainable and 896 are not.

We trained the model using 10 epochs ,each epoch having 100 steps and 25 validation steps.

The model gave an accuracy level of 78.37% on the test set and 91.49% on the train set.

![image](https://github.com/user-attachments/assets/07a6d9bd-7d4b-4b43-bf99-337c20bab789)


#### 2. DENSENET

The model has  7,037,504 total parameters where 6,953,856 are trainable and 83,648 are not.

We trained the model using 10 epochs ,each epoch having 100 steps and 25 validation steps.

The model gave an accuracy level of 76.92% on the test set and 82.69% on the train set.

![image](https://github.com/user-attachments/assets/61137816-6238-4df2-bd39-8476267ac1bd)



#### 3. VGG16

Presented in 2014, VGG16 has a very simple and classical architecture, with blocks of 2 or 3 convolutional layers followed by a pooling layer, plus a final dense network composed of 2 hidden layers (of 4096 nodes each) and one output layer (of 1000 nodes). Only 3x3 filters are used.

We trained the model using 10 epochs ,each epoch having 100 steps and 25 validation steps.

The model gave an accuracy level of 46.63% on the test set and 38.02% on the train set,which was dismal performance.

![image](https://github.com/user-attachments/assets/5d5b6e4d-1a14-403e-9cb6-3e031a5eaac4)


#### 4.ResNet

See the full explanation and schemes in the Research Paper on Deep Residual Learning (https://arxiv.org/pdf/1512.03385.pdf)

The model has  23,587,712 total parameters where 23,534,592 are trainable and 53,120 are not.

We trained the model using 10 epochs ,each epoch having 100 steps and 25 validation steps.

The model gave an accuracy level of 63.30% on the test set and 74.98% on the train set

![image](https://github.com/user-attachments/assets/963ba40d-fcea-4a06-bbf8-75e54c54bc09)


#### 5. InceptionNet

Also known as GoogleNet, this architecture presents sub-networks called inception modules, which allows fast training computing, complex patterns detection, and optimal use of parameters

for more information visit https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43022.pdf


We trained the model using 10 epochs ,each epoch having 100 steps and 25 validation steps.

The model gave an accuracy level of 79.17% on the test set and 89.57% on the train set

![image](https://github.com/user-attachments/assets/be43078a-5fc5-4e4c-9cfa-90a615fdc564)


#### COMPARING THE DIFFERENT MODEL PERFORMANCE

![image](https://github.com/user-attachments/assets/b1563476-8f71-4f7e-8831-c880f88a8271)


#### OBSERVATIONS

1) We will deploy our best performing model,the InceptionNet.

2) By using pretrained models like ResNet, VGG, and  DenseNet and fine-tuning them  resulted in better performance compared to training models from scratch.

3) Models trained on specific datasets may not generalize well to data from different hospitals or demographic groups. It’s important to validate models on diverse datasets to ensure they are robust and generalizable.

####  RECOMMENDATIONS

1) Integrate with Existing Systems: Ensure the model can be seamlessly integrated into existing hospital IT systems and clinical workflows to enhance usability.

2) Optimize for Real-Time Use: Develop the model to provide real-time predictions, aiding clinicians in making timely and accurate diagnoses.

3) Implement Continuous Learning: Establish a process for ongoing data collection, model retraining, and validation to keep the model updated with new information and medical advancements.

4) Involve Clinicians in Development: Collaborate closely with radiologists and other healthcare providers to ensure the model meets clinical needs and addresses practical challenges.

5) Ensure Accurate Labeling: Engage experienced radiologists to accurately annotate chest X-ray images, ensuring reliable ground truth for training the model.

6) Align with Strategic Goals: Ensure the project aligns with the organization’s strategic objectives, such as improving diagnostic accuracy, reducing healthcare costs, and enhancing patient outcomes.

####  CONCLUSION

In conclusion, our best model did very well at identifying the pneumonia images with 79.17 % accuracy and 93.3% recall. We care most about the recall score because this represents how well our model does at predicting the class of images belonging to those with pneumonia. Given our business case, false negatives (patient has pneumonia but we classify them as healthy) are much more dangerous than false positives (patient is healthy but we classify them as having pneumonia).

Meanwhile, we can keep tuning our model based on the feedback in order to improve our recall. Plus, there are many pre-trained neural network models already available online, so we can run our data on those models and make some improvements to our model by comparing our model with those pre-trained models. Lastly, we don't have strong background knowledge on identifying Pneumonia from an X-Ray image, so it is necessary to tackle this issue with domain experts and consult them on questions such as, what features in an X-Ray image that our model misclassified could a radiologist point out to modify our model.




