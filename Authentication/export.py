import tensorflow as tf
from model import L1Dist
from model import make_siamese_model
import os

export_path = "siamesemodelv2_keras"

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


siamese_model.save(export_path, save_format="tf")

print(f"Model exported to {export_path}")
