# MNIST-Neural-Net-Classifier
## How to use
Run Display.py and wait for the model to train. Once the GUI appears, draw different digits and press Enter to see models prediction. To reset GUI window press 'r'.

## Changing the model
Model hyperparameters can be updated in the Display.py file as well - number of iterations and learning rate can be modified in the call to generate_model(). Change the structure of the Neural network by modifying the global variable "layers_dims", and choose the activation functions by changing "layer_activation_list". 

## Known bugs
While model is around 90% accurate on MNIST test set data, model sometimes struggles with classification from the GUI. Try to keep handwriting neat and clear for the best classification results.
