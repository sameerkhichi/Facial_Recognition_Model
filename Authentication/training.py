import tensorflow as tf
import os
from model import make_siamese_model
from tensorflow.keras.metrics import Precision, Recall
import numpy as np
from data_preprocessing import train_data

#create the network
siamese_model = make_siamese_model()

#loss function binary crossentropy
binary_cross_loss = tf.losses.BinaryCrossentropy()

#using an adam optimizer to adjust internal parameters (weights and biases) to minimize loss function
#adam was used for ease of use and fast convergence 
opt = tf.keras.optimizers.Adam(1e-4) #0.0001 learning rate back propagation

'''
To reload from the checkpoints you can use model.load(path to checkpoint) .
This will load the pre trained weights into the existing model 
'''
#establishing training checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

'''
This flow is used for any type of neural network
-Once batch of data goes through to the training step
-make a prediction
-calculate loss
-calculate gradients
-apply back prorogation (calculate new weights)through neural network to achieve best possible model
'''

#train step function
@tf.function #tf function decorator - compiles function into a callable tensorflow graph
def train_step(batch):
    
    #record all of our operations for automatic differentiation 
    with tf.GradientTape() as tape:
        #get anchor and positive/negative image
        x = batch[:2] #extracting the batch of images passed in
        #get label
        y = batch[2]

        #forward pass
        y_prime = siamese_model(x, training=True) #this is a prediction
        
        #calculate loss
        loss = binary_cross_loss(y, y_prime) #passing in true and predicted value to find loss

    #calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables) #grabs loss and all gradiants with loss in terms of all trainable variables from tape

    #calculate updates weights and apply to the siamese model using adams optimization algorithm(back propagation)
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss

#This function basically loops over every batch in the dataset rather than focussing on one batch like the train step function
#EPOCHS is one complete pass of the entire training dataset through the neural network
def train(data, EPOCHS):
    
    #loop through the EPOCHS
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data)) #for a progress bar

        #creating a metric object
        r = Recall()
        p = Precision()

        #Loop through each batch of data
        for idx, batch in enumerate(data):

            #running training step here
            loss = train_step(batch) #returns loss
            y_prime = siamese_model(batch[:2], training=False)

            r.update_state(batch[2], y_prime)
            p.update_state(batch[2], y_prime)

            progbar.update(idx+1, [("loss", loss)])

        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        
        #saving a checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


#Run model training
EPOCHS = 141
train(train_data, EPOCHS)