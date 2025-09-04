import tensorflow as tf
from model import make_siamese_model
from model import L1Dist
import os

siamese_model = make_siamese_model()

#restoring checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(siamese_model=siamese_model)

#restore from specific checkpoint
checkpoint_to_restore = os.path.join(checkpoint_dir, "ckpt-15")
if tf.io.gfile.exists(checkpoint_to_restore + ".index"):
    checkpoint.restore(checkpoint_to_restore).expect_partial()
    print(f"Restored model from {checkpoint_to_restore}")
else:
    print("!!!!Specified checkpoint not found, evaluating untrained model!!!!")

'''
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint).expect_partial()
    print(f"Restored model from {latest_checkpoint}")
else:
    print("!!!!No checkpoint was found, evaluating untrained model!!!!")
'''

siamese_model.save('siamesemodelv2.h5')

#reload model
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

print("-----------------------------------------")
print(siamese_model.input_shape)
print(siamese_model.output_shape)
print("-----------------------------------------")

siamese_model.summary()