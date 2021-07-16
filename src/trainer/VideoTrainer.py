import tensorflow as tf
from utils import initializer
from .base_trainer import BaseTrainer


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(WarmupSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class VideoTrainer(BaseTrainer):

    @initializer
    def __init__(self, model, config, train_ds=None, val_ds=None):
        learning_rate = WarmupSchedule(self.config['trainer']['lr_dmodel'])
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')

        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            'train_accuracy')
        self.test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            'test_accuracy')
        super(VideoTrainer, self).__init__(model, config, optimizer)

    def _train_epoch(self, epoch):
        pass

    def test(self):
        pass
