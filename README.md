# Biomedical-Image-Analysis
**Data synthesis and segmentation for brain scans with midline shift**

## Getting started

### Environment setup

An Anaconda virtual environment for this project is not required but highly recommended:

```commandline
conda create -n mls python=3.9
conda activate mls
```

All required Python packages are listed in requirements.txt. Due to the nature of PyTorch, please do not directly use
pip to install all required packges. Instead, go to [PyTorch's website](https://pytorch.org/get-started/locally/) to 
install PyTorch first. Depending on what platform is used, it should provide a command line code like this:

```commandline
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Then it's safe to run

```commandline
pip install requirements.txt
```

to install all other required packages.

### Load Data

There's a small dataset already given in /Data/JPG and /Data/PNG, and their resized version in /Data/Resized. 

![](/Data/JPG/26.jpg)

If you wish to use your own dataset, please make sure that they are in the same format and stored under a new folder 
under /Data. The program will automatically resize your data and give you the option to store them. However,
numerically ordered data names are preferred.

If your data are not in regular image format or are not uniformly shaped, there's some helper function in *data.py*
that could be useful, like *toPNG()* or *expandData()*.

## Running the project

### Entire project

To run the entire project, which includes data resizing and transformation, model training, and visualizations, 
first change the global variables in *main.py* to their correct settings and run:

```commandline
python main.py
```

Shift magnitude (in millimeter) could be manually changed in the script or provided as an additional argument. 
For example, this runs the project with a 5mm shift:

```commandline
python main.py 5
```

Currently this project only accept shifts in integers so every float input would be rounded.

### Transformation

To run the data transformation section of the project, change the global variables in *transform.py* to their
correct settings and run:

```commandline
python transform.py
```

There's many data input and file saving options provided from the functions in *transform.py* that could be helpful.

In addition, additional methods of transformation could be created and customized by creating a new class 
that inherits class *Method*:

```python
class NewMethod(Method):

    def __init__(self):
        super().__init__("Method Name")
        # additional variables

    def transform(self, img, shift):
        # customized image transform method
        return img

    def mask(self, img, shift):
        # customized mask (left brain segmentation) for the transform method above
        return mask
```

Like *main.py*, shift magnitude could also be provided as an additional argument:

```commandline
python transform.py 5
```

### Training

It is recommended to run data transformation section before running the model training section of the project.
Specifically, model training section loads training X-values and training Y-values from (using shift=5mm as example) 
/Data/s5trainX.npy and /Data/s5trainY.npy, files saved by the data transformation scripts. 

If the numpy files already exist (shift with 5mm and 10mm are provided with the project), use 

```commandline
python train.py
```

to deploy and train the model. The model will be stored in /Model and the average dice score from the dataset 
will be printed. In default, performance of the model during training process will also be visualized and 
saved in /Model

Like *main.py*, shift magnitude could also be provided as an additional argument:

```commandline
python train.py 5
```

## Visualization

Many images are created while running the entire project:

### Training set

Since the original dataset is not guaranteed to be model-ready, after resizing the original data to get the training 
set, every image that is about to go through the model is saved (with axis and labels) to /Data/Resized.

![](/Data/Resized/26.jpg)

### Mask

Based on the transformation method, every image is transformed and the actual mask, i.e. Y-value of the training set, 
is projected and saved to /Data/Mask for references.

![](/Data/Mask/s10-26.jpg)

### Model Performance

As mentioned earlier, the performance of model during the entire training process is saved into two png files
(using shift=5mm as example) named Loss-s5 and Accuracy-s5 after the model is trained.

![](/Model/Loss.png)

![](/Model/Accuracy.png)

### Result Comparison

Once the model is trained, the result, or the predicted mask, is obtained and for each of the images in the training
set, a visualization is created where the original image, the actual mask and the predicted mask are plotted 
side by side and saved to /Data/Comparison.

![](/Data/Comparison/s10-26.png)

## Troubleshooting

If PyTorch and Tensorflow could not work together within a single environment or there's import issues, please try
creating two separate environment, one with PyTorch and one with Tensorflow to run training section. 
First use the environment with PyTorch to run the transformation section, saving intermediate files to local, 
and then use the environment with Tensorflow to run the training section to get the final results.
All other required packages should be installed on both environments.

## Presentation

The slides used in course presentation for 16-725 Methods In (Bio)Medical Image Analysis at Carnegie Mellon University
could be found in /Presentation.

## References

Kaggle [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) Dataset
