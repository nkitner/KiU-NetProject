import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

# Load saved models
unet_reconstructed = keras.models.load_model('/content/unet_model_h5.h5')
kiunet_reconstructed = keras.models.load_model('/content/kiUnet_model2_3d_a.h5')



# Evaluate KiU-Net
## Instantiate logging for tensorboard
kiunet_eval_log_dir = "logs/kiunet/eval/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
kiunet_eval_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=kiunet_eval_log_dir, histogram_freq=1)

kiunet_reconstructed.evaluate(x=x_test, y=y_test, batch_size=1,
                              verbose=1,
                              callbacks=[kiunet_eval_tensorboard_callback])

# Evaluate U-Net
## Instantiate logging for tensorboard
unet_eval_log_dir = "logs/unet/eval/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
unet_eval_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=unet_eval_log_dir, histogram_freq=1)

unet_reconstructed.evaluate(x=x_test, y=y_test, batch_size=1,
                              verbose=1,
                              callbacks=[unet_eval_tensorboard_callback])