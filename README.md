# automated_waste_seg
A deep learning-based approach for classifying waste into organic and recyclable categories. The model integrates VGG16 for image classification via transfer learning and YOLOv12 for object detection. 
The resulting model, named SORT (Sorting of Organic and Recyclable Trash), was trained using TensorFlow and Keras, achieving an accuracy of 94% and a loss of 0.365.
Deployment was achieved through a Flask-based web application, allowing users to upload images and receive real-time predictions with confidence scores. The user interface was designed for ease of use, featuring modern elements like animated progress bars, and success indicators.
