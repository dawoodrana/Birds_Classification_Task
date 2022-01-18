# **Birds_Classification_Task**

## Steps to run the respective task code

1. Download the [Dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
2. Try to run the code on GPU server
3. Google colab is used to perform all the tasks
4. Mount your drive or cloud drive with the repective folder where you downloaded the data
5. Run all the respective installation first which are already provided in the codes
6. All the necessary **imports** are given in the codes, run them all
7. **Important**, in Task 2 -> 'Train and Test Data split' code, when splitting is done, data from 'image' folder is splitted into 'test' folder, now this 'image' folder have remaining train data, so rename it as 'train' i.e., 'image' -> 'train'
8. When ever you are running the that part of code, where 'file' is required like **'train', 'test', 'images','classes.txt'** etc. Make sure give them the respective path where these files are located
9. All the detail of performances are mentioned in Tasks
10. Now run the code Task wise, as given below



## Task 2
**1. Dataset**

[Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

1- Number of classes: 200

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


**2. Train and Test Data split**

* 'train_test_split.txt' ( Splitting the data on the basis of this text file, provided in dataset i.e., '0' for test_data and '1' for train_data)
* {'test': 5794, 'train': 5994}

**Respective google colab code is attached:**

* [Code for Train and Test data splitting](https://colab.research.google.com/drive/1gUwr7VdE4gw7YUmeTp1KrQj5HwG51sJM#scrollTo=jeDmgESr7TOH)

**3. Exploratory Data Analysis**

* Required Installation

```bash
!pip install hvplot
```

* Loading and Normalization of Data
* Visualization and Skimming through some samples
* Class Sampling

  * Plotting the class samples

* Checked Image size variability
  
  * Heigth and width using boxplot

**Respective google colab code is attached:**

* [Code for EDA](https://colab.research.google.com/drive/17vOxQgpBOllKFrZ0n67FlNYt3HrfYymv#scrollTo=ITgHOKTm4hZM)


## Task 3

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

**Required google colab code is attached:**

[Code for Training data on pre-trained model and features extraction](https://colab.research.google.com/drive/1FZo28vtsq_wPdgVg0emBtpi5IiQE8NAI#scrollTo=ryJ5vtkVXhRo)

## Task 4

**Loading Saved Model from numpy file**

point_resnet_best.npy

**1. Metices used for evaluation of Image Retrieval models**

* Finding Accuracy at k= 1 and k= 5

Here I have defined a use full class and a function to find top-1 and top-5 accuracies

**2.  Information for listed metrics**

Accuracy at k=1: 69.5029 %

Accuracy at k=5: 91.4912 %

**3.  Train and test features used to compute evaluation metrics**

**4. Compute the Evaluation Metrices**

Once I have got images in the correct format. I have written a function for making predictions with my model. A common practice is to predict the top 5 or so (usually called top- K ) most probable classes. I have calculated the class probabilities then find the K largest values.
To get the top K largest values in a tensor I have used x.topk(k). This method returns both the highest k probabilities and the indices of those probabilities corresponding to the classes.

Now I have used a trained model for predictions. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. I have used matplotlib to plot the probabilities for the top 5 classes as a bar graph, along with the input image.

**Required google colab code for this task is attached:**

[Code for Evaluation Metrices](https://colab.research.google.com/drive/1KuJT-eVeO4ZVRAaOXnOwB876veUBgPkt#scrollTo=YtftovmWU15m)





