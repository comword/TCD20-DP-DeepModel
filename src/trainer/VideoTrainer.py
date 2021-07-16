import tensorflow as tf
from tqdm import tqdm
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

    def __init__(self, model, config, data_loader):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        if data_loader.hasSplitValidation():
            self.train_ds, self.val_ds = data_loader.getSplitDataset()
            self.do_validation = True
        else:
            self.do_validation = False
        learning_rate = WarmupSchedule(self.config['trainer']['lr_dmodel'])
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        super(VideoTrainer, self).__init__(model, config, optimizer)

    def _train_epoch(self, epoch):
        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            'train_accuracy')
        tbar = tqdm(self.train_ds, total=self.data_loader.getTrainLen())
        for batch_idx, (imgs, frame_idx, target) in enumerate(tbar):
            batch_size = imgs.shape[0]
            with tf.GradientTape() as tape:
                predictions = self.model([imgs, frame_idx], training=True)
                loss = self.loss(target, predictions)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

            train_loss(loss)
            train_accuracy(target, predictions)
            self.writer.set_step((epoch - 1) * batch_size + batch_idx)
            tbar.set_description('Train loss: %.3f' % train_loss.result())
        loss_res = train_loss.result()
        acc_res = train_accuracy.result()
        self.writer.scalar('train/loss', loss_res, epoch)
        self.writer.scalar('train/accuracy', acc_res, epoch)

        log = {
            'train_loss': loss_res,
            'train_accuracy': acc_res
        }
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})
        return log

    def _valid_epoch(self, epoch):
        test_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            'val_accuracy')
        tbar = tqdm(self.val_ds, total=self.data_loader.getValLen())
        for batch_idx, (imgs, frame_idx, target) in enumerate(tbar):
            predictions = self.model([imgs, frame_idx], training=False)
            loss = self.loss(target, predictions)

            test_loss(loss)
            test_accuracy(target, predictions)
            tbar.set_description('Test loss: %.3f' % test_loss.result())
        loss_res = test_loss.result()
        acc_res = test_accuracy.result()
        self.writer.scalar('val/loss', loss_res, epoch)
        self.writer.scalar('val/accuracy', acc_res, epoch)
        return {
            'val_loss': loss_res,
            'val_accuracy': acc_res
        }

    # TODO: full test
    def test(self):
        pass
