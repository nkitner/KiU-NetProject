'''This file holds the training loops for the KiU-Net model found in model.py

Class: CISC867-Deep Learning
Author: Henry Lee
Date: 2021-10-07
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time    # records the computation time
import model as kiunet

### CONVERT TO STATIC GRAPH FOR TF.FUNCTION, REMOVES EAGER EXECUTION AND IMPROVES SPEED

# Instantiate optimizer
optimizer = keras.optimizers.Adam(learning_rate=0.001)    ## learning_rate = 0.001 for KiU-Net and 0.0001 for 3D

# Instantiate loss function
loss_fn = keras.losses.binary_crossentropy(from_logits=False)    ## maybe set as true?

# Instantiate evaluation metrics
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Instantiate training loop
epochs = 400
batch_size = 1
model = kiunet.kiunet((128, 128, 3))

for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over batches of the dataset
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open GradientTape to record the operations run
        # during forward pass  ==> allows auto-differentiation
        with tf.GradientTape() as tape:

            # Run forward pass of layer
            # Operations that layer applies to its inputs
            # are recorded on GradientTape
            logits = model(x_batch_train, training=True)

            # Compute loss value for minibatch
            loss_value = loss_fn(y_batch_train, logits)
        
        # Use GradientTape to retreive gradients of trainable
        # variables with respect to loss
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating the value 
        # of variables to minimze the loss
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Update training metric
        train_acc_metric.update_state(y_batch_train, logits)

        # Log every 200 batches
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * batch_size))
    
    # Display metrics at end of each epoch
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run validation loop at end of each epoch
    for x_batch_val, y_batch_val in val_dataset:
        val_logits = model(x_batch_val, training=False)
        # Update val metrics
        val_acc_metric.update_state(y_batch_val, val_logits)
    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %2.fs" % (time.time() - start_time))