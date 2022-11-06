# Video-based-Action-Recognition

## Summary

In this repository, a video based action recognition model is implemented 

## About the dataset 

The dataset used is the **kinetics400** dataset which is a collection of action videos collected from Youtube. The dataset consists of trimmed clips and mainly focuses on single person action classifications.

## About the model 

The model used is the **inflated 3D inception** model which has been pre-trained on the kinetics400 dataset. This model is chosen because it is able to reuse the 2D CNN's model architecture for higher accuracy in classifications.

## Requirements

First, run the following 2 lines to install the required libraries

```
pip install --upgrade mxnet

pip install --upgrade gluoncv
```

## About the repository 

**exploration.ipynb** shows an experimentation with the model as seen [here](https://cv.gluon.ai/build/examples_action_recognition/demo_i3d_kinetics400.html). By testing it on a stock video of someone smoking (**inferencevideo.mp4**), the model is able to predict the smoking activity. 

**inference.py** is an attempt at a real-time application of the same model by invoking the webcam. 

### Notes
The application is not exactly "real-time" as model prediction takes about 40 seconds. However, by making an initial pose (Eg. yawning) when running the script, the prediction was actually able to make a prediction for the correct activity.

## Try it out

To run the inference, clone this repository and run the following line

```
python inference.py
```
