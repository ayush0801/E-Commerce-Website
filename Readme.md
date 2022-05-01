# Detection of Melanoma

Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective.\
\
Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery. Image analysis tools that automate the diagnosis of melanoma will improve dermatologists' diagnostic accuracy. Better detection of melanoma has the opportunity to positively impact millions of people. 

Our model predicts a binary target for each image. It predicts the probability (floating point) between 0.0 and 1.0 that the lesion in the image is malignant (the target). In the training data, train.csv, the value 0 denotes benign, and 1 indicates malignant.

## Files
* train.csv - the training set
* test.csv - the test set

## Columns
* image_name - unique identifier, points to filename of related DICOM image
* patient_id - unique patient identifier
* sex - the sex of the patient (when unknown, will be blank)
* age_approx - approximate patient age at time of imaging
* anatom_site_general_challenge - location of imaged site
* diagnosis - detailed diagnosis information (train only)
* benign_malignant - indicator of malignancy of imaged lesion
* target - binarized version of the target variable

## Dataset
The dataset which has been used for the project is collected from a [Kaggle Competition](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data?select=train).

## Plotted images
1. is malignant images, 
2. is benign images,
3. inpainted images(after hair removal)
4. anatom_site_general_challenge pi chart,
5. count of age approx, 
6. count of diagnosis and many more in the notebooks.


## Model structure for VGG:
```python
model.add(VGG19(include_top=False, weights='imagenet', input_shape= inputShape))
model.add(Flatten())
model.add(Dense(32))
model.add(LeakyReLU(0.001))
model.add(Dense(16))
model.add(LeakyReLU(0.001))
model.add(Dense(1, activation='sigmoid'))
```
## ResNet101 Model summary:
```python
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
resnet101 (Model)            (None, 7, 7, 2048)        42658176  
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 1)                 2049      
=================================================================
Total params: 42,660,225
Trainable params: 42,554,881
Non-trainable params: 105,344
_________________________________________________________________

``` 

## File structure:
1) ResNet101 training inside `resnet101-with-focal-loss-and-img-aug.ipynb`
2) VGG16 training is in `baseline-submission-keras-vgg16`
3) Image Analysis is in `cancer-detection-analysis.ipynb` 
4) Tabular data analysis is in `eda-w-plotly-and-stacking-on-tabular-data-0-685.ipynb`

