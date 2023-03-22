# Python-Optical-Character-Recognition
An Optical Character Recognition application developed in Python based on the Keras, and OpenCV libraries

# Prerequisite Libraries:
<ul type="-">
<li url="https://www.tensorflow.org/datasets/overview">tensorflow-datasets</li>
<li url="https://www.tensorflow.org/install">tensorflow</li>
<li url="https://matplotlib.org/stable/users/installing/index.html">matplotlib</li>
<li url="https://pypi.org/project/opencv-python/">opencv-python</li>
</ul>

# How to use:
1. In the main function, the mode could be select either "image" or "webcam" where image will use the image_file variable as the filepath and recognize the image, webcam will connect to the device camera, and use the captured live video to perform recognition<br>

2. The file variable indicate the path to store pre-trained model, where if there is pre-trained model, the application will skip the model training process which taken a long time for pre-processing and model training<br>

3. Parameter Tuning:<br>
preprocess_data function: resize of image data and train test split percentage<br>

train_model function:<br>
batch_size and epochs of the CNN<br>

build_cnn function:<br>
network structure, number of filter, filter size, activation functions of CNN<br>


# Implementation Steps:
<h2>Model Training</h2>
1. Extended MNIST dataset[1] with handwritten digit and alphabets is loaded from tensorflow datasets[2]<br>
2. As the images are inverted horizontally and rotated 90 deg anti-clockwise[2], the images are transposed by using map function provided by tensorflow datasets.<br>
3. The original image data with size of [0,255] for each bit is then shaped and convert into range of [0,1]<br>
4. The number of unique output class label in then extracted from dataset<br>
5. Dataset is splited into 80% training data and 20% testing data<br>
6. A Sequential CNN is built and the model is trained<br>
7. The trained results are plot using matplotlib libraries to inspect model performance [3]<br>

<h2>Image Recognition</h2>
1. The image file is loaded, and converted into grayscale image<br>
2. The image is blurred to reduce the noise<br>
3. Canny filter is used to extract the edge of handwritten characters<br>
4. findContours is used to filter and list out the contours for each characters<br>
5. the contours are sorted from left-to-right, top-to-bottom [4]<br>
6. for each contours, if the size within pre-defined range, the character is extracted from blurred image, and thresholded.<br>
7. thresholded image is then resized [4] and border is added to the size of image<br>
8. Each thresholded image is then feed to the pre-trained model and the most likely outcome (highest percentage) will be labelled in the image along with the contour block [4]<br>



# References:
<ol type="1">
<li>[1] NIST. The EMNIST Dataset [online]. Available at: https://www.nist.gov/itl/products-and-services/emnist-dataset (Accessed: 22 March 2023)</li>
<li>[2]TensorFlow. emnist [online]. Available at: https://www.tensorflow.org/datasets/catalog/emnist (Accessed: 22 March 2023)</li>
<li>[3] Muralidhar, K. (2021). Learning Curve to identify Overfitting and Underfitting in Machine Learning | Towards Data Science [online]. Available at: https://towardsdatascience.com/learning-curve-to-identify-overfitting-underfitting-problems-133177f38df5 (Accessed: 22 March 2023)</li>
<li>[4] Rosebrock, A. (2020). OCR: Handwriting recognition with OpenCV, Keras, and TensorFlow | PyImageSearch [online]. Available at: https://pyimagesearch.com/2020/08/24/ocr-handwriting-recognition-with-opencv-keras-and-tensorflow/ (Accessed: 22 March 2023)</li>
</ol>
