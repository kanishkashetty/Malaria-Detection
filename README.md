# Malaria-Detection-Deep Learning
Image classification using Convolutional Neural Networks (CNNs) with two different architectures: VGG19 , simple Sequential model.

**VGG19 Model:**
Import ibraries from TensorFlow/Keras for building and training neural networks.
Used VGG19 as a base model with pre-trained ImageNet weights and excluded the top layer (include_top=False).
Froze the layers of the VGG19 model to prevent their weights from being updated during training.
Added a Flatten layer followed by a Dense layer with softmax activation for the final classification.
Created the model and compiled it using categorical cross-entropy loss and the Adam optimizer.
Used an ImageDataGenerator to perform data augmentation on the training images.
Trained the model using fit_generator for 50 epochs.
Saved the model and made predictions on the test set.
**Simple Sequential Model:**
Imported the required libraries.
Created a Sequential model with Conv2D, MaxPooling2D, Flatten, and Dense layers.
Compiled and trained this model similarly to the VGG19 model using the ImageDataGenerator.
