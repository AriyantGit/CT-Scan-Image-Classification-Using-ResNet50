# Covid---19-CT-Scan

COVID-19 Detection based on Chest X-rays and CT Scans using four Transfer Learning algorithms: VGG16, ResNet50, InceptionV3, Xception. The models were trained for 500 epochs on around 1000 Chest X-rays and around 750 CT Scan images on Google Colab GPU. After training, the accuracies acheived for the model are as follows:
<pre>
                ResNet50
CT Scans        89%      
</pre>
A Flask App was later developed wherein user can upload Chest  CT Scans and get the output of possibility of COVID infection.

The article for the project was selected and published in <b>Towards Data Science</b>:<br> 
https://towardsdatascience.com/covid-19-detector-flask-app-based-on-chest-x-rays-and-ct-scans-using-deep-learning-a0db89e1ed2a

# Dataset
The dataset for the project was gathered from two sources:
CT Scan images (2480 Image files belonging to 2 classes) were obtained from: https://shorturl.at/ceCIQ
80% of the images were used for training the models and the remaining 20% for testing

# Technical Concepts
<b>ImageNet</b> is formally a project aimed at (manually) labeling and categorizing images into almost 22,000 separate object categories for the purpose of computer vision research.<br>
More information can be found <a href="https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/">here</a>
<br>
<br>
<b>ResNet50</b>ResNet-50 is a convolutional neural network (CNN) architecture that has 50 layers. It was introduced by Microsoft Research in 2015 and has become one of the widely used models for various computer vision tasks, including image classification, object detection, and image segmentation.

The name "ResNet" stands for "Residual Network," which refers to the use of residual connections or skip connections in the network. These connections enable the network to learn residual mappings, which help in addressing the degradation problem that often occurs when training deep neural networks.<br>
<ul>
  <li>Input: The network takes an input image of size 224x224 pixels with three color channels (RGB).</li>
  <li>Convolutional Layers: The input image goes through a sequence of convolutional layers, where each layer applies a set of filters to extract features from the input. ResNet-50 uses various filter sizes, such as 1x1, 3x3, and 1x1, to capture different levels of information.</li>
  <li>Skip Connections: ResNet-50 introduces skip connections that allow the network to directly pass information from one layer to a later layer, bypassing a few intermediate layers. </li>
  <li>Global Average Pooling: After the convolutional layers, a global average pooling layer is applied, which reduces the spatial dimensions of the features to a fixed size, regardless of the input image size. This step helps in reducing overfitting.</li>
  <li>Fully Connected Layer: Finally, a fully connected layer is applied to the pooled features to perform the classification task. For ResNet-50, this layer typically consists of 1,000 neurons, corresponding to the number of classes in the ImageNet dataset, which the model was originally trained on.</li>
  <li>Output Layer Softmax Activation: The output of the classification layer is often passed through a softmax activation function, which converts the class probabilities into a normalized distribution, ensuring that the probabilities sum up to 1. This allows for easier interpretation and decision-making.</li>
</ul>
![Alt text](https://www.google.com/url?sa=i&url=https%3A%2F%2Fstackoverflow.com%2Fquestions%2F54207410%2Fhow-to-split-resnet50-model-from-top-as-well-as-from-bottom&psig=AOvVaw1Mn8Sz6exWkqfLN2wHrEPo&ust=1686499729815000&source=images&cd=vfe&ved=0CBEQjRxqFwoTCODG3Z2Luf8CFQAAAAAdAAAAABAJ)
More information can be found <a href="https://www.mathworks.com/help/deeplearning/ref/resnet50.html#:~:text=ResNet%2D50%20is%20a%20convolutional,%2C%20pencil%2C%20and%20many%20animals.">here</a>
<br>

## How to use Flask App
<ul>
  <li>Download repo, change to directory of repo, go to command prompt and run <b>pip install -r requirements.txt</b></li>
  <li>The dataset and models of the repository have been moved to Google Drive due to expiry of my Github LFS. So please download the zip file from <b><a href="https://drive.google.com/file/d/1dA-rdmDmCGa3xxW5KpfLJdo7M54lPcQq/view?usp=sharing">here</a></b>, extract it and replace the above data and models folder with these. Also make sure you have <b>PYTHON V 3.8.5</b>. Other versions might not be supported</li>
  <li>On command prompt, run <b>python app.py</b></li>
  <li>Open your web browser and go to <b>127.0.0.1:5000</b> to access the Flask App</li>
</ul>

## How to use Jupyter Notebooks 
<ul>
  <li>Download my repo and upload the repo folder to your <b>Google Drive</b></li>
  <li>Go to the jupyter notebooks folder in my repo, right click the notebook you want to open and select <b>Open with Google Colab</b>   </li>
  <li>Activate free <b>Google Colab GPU</b> for faster execution. Go to Runtime -> Change Runtime Type -> Hardware Accelerator -> GPU -> Save</li>
</ul>