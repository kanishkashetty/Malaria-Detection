# Malaria-Detection-Deep Learning
Image classification using Convolutional Neural Networks (CNNs) with two different architectures: VGG19 , simple Sequential model.

**VGG19 Model:**
1. Import ibraries from TensorFlow/Keras for building and training neural networks.
2. Used VGG19 as a base model with pre-trained ImageNet weights and excluded the top layer (include_top=False).
3. Froze the layers of the VGG19 model to prevent their weights from being updated during training.
4. Added a Flatten layer followed by a Dense layer with softmax activation for the final classification.
5. Created the model and compiled it using categorical cross-entropy loss and the Adam optimizer.
6. Used an ImageDataGenerator to perform data augmentation on the training images.
7. Trained the model using fit_generator for 50 epochs.
8. Saved the model and made predictions on the test set.
   
**Simple Sequential Model:**
1. Imported the required libraries.
2. Created a Sequential model with Conv2D, MaxPooling2D, Flatten, and Dense layers.
3. Compiled and trained this model similarly to the VGG19 model using the ImageDataGenerator.
Finally, Flask web application to perform image classification using a pre-trained VGG19 model.
