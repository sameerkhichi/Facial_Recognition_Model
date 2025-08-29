import tensorflow as tf
from model import make_siamese_model
from model import L1Dist

siamese_model = make_siamese_model()

#restoring checkpoint
checkpoint_dir = './training_checkpoints'
checkpoint = tf.train.Checkpoint(siamese_model=siamese_model)
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    checkpoint.restore(latest_checkpoint).expect_partial()
    print(f"Restored model from {latest_checkpoint}")
else:
    print("!!!!No checkpoint was found, evaluating untrained model!!!!")


siamese_model.save('siamesemodelv2.h5')

#reload model
siamese_model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

print("-----------------------------------------")
print(siamese_model.input_shape)
print(siamese_model.output_shape)
print("-----------------------------------------")

siamese_model.summary()