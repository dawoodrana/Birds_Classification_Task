# **Birds_Classification_Task**

# Task 2
**1. Dataset**

[Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

1- Number of categories: 200

2- Number of images: 11,788

Download this dataset to either to your google drive or cloud



**Requirments**

 * Python 3.8v
 * Pandas
 * NumPy
 * PyTorch
 * Matplotlib
 * hvplot
 * seaborn
 * os


**2. Exploratory Data Analysis**

* Loading and Normalization of Data
* Visualization and Skimming through some samples
* Class Sampling

  * Plotting the class samples

* Checked Image size variability
  
  * Heigth and width using boxplot

**3. Train and Test Data split**

* 'train_test_split.txt' ( Splitting the data on the basis of this text file, provided in dataset i.e., '0' for test_data and '1' for train_data)

Respective google colab codes for this task;

* [Code for EDA](https://colab.research.google.com/drive/17vOxQgpBOllKFrZ0n67FlNYt3HrfYymv#scrollTo=ITgHOKTm4hZM)
* [Code for Train and Test data splitting](https://colab.research.google.com/drive/1gUwr7VdE4gw7YUmeTp1KrQj5HwG51sJM#scrollTo=jeDmgESr7TOH)

# Task 3

Installation of PyTorch on Google Colab

```bash
!pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl 

!pip3 install torchvision
```

**Loading and Normalization of Data**

We will use torchvision and torch.utils.data packages for loading the data. For the training, I have applied transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. I also made it sure that the input data is resized to 224x224 pixels as required by the pre-trained networks.

The testing set is used to measure the model's performance on data it hasn't seen yet. For this I have not performed any scaling or rotation transformations, but I had resized and then cropped the images to the appropriate size.

The pre-trained network I have used was trained on the ImageNet dataset where each color channel was normalized separately. For all three sets i have normalized the means and standard deviations of the images to what the network expects. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225], calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range from -1 to 1.

**Visualize a few images**

**Train and saving model in 'numPy' and 'pytorch'**

Features extracted from the pretrained  modal are save in 'numpy' -> 'np.save' for next task


**Fetching the features using Pytorch pre-trained modal**

Load a pretrained Resnet 18 model and reset final fully connected layer.
   * Pre-trained Resnet_18 model is used

**Train the model on 50 epochs**

Required google colab code is attached:

[Code for Training data on pre-trained model and features extraction](https://colab.research.google.com/drive/1FZo28vtsq_wPdgVg0emBtpi5IiQE8NAI#scrollTo=ryJ5vtkVXhRo)






