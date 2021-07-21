import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from .base_trainer import BaseTrainer


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=2000):
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
            self.train_ds, self.val_ds = data_loader.getFullDataset(), None
            self.do_validation = False
        learning_rate = WarmupSchedule(self.config['trainer']['lr_dmodel'],
                                       warmup_steps=self.config['trainer']['warmup_steps'])
        # optimizer = tf.keras.optimizers.Adam(learning_rate)
        optimizer = tfa.optimizers.SGDW(
            learning_rate=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
        self.train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            'train_accuracy')
        self.test_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            'val_accuracy')
        super(VideoTrainer, self).__init__(model, config, optimizer)

    def _train_epoch(self, epoch):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        tbar = tqdm(self.train_ds, total=self.data_loader.getTrainLen())
        for batch_idx, (imgs, frame_idx, target) in enumerate(tbar):
            with tf.GradientTape() as tape:
                predictions = self.model([imgs, frame_idx], training=True)
                loss = self.loss(target, predictions)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

            self.train_loss(loss)
            self.train_accuracy(target, predictions)
            step_count = (epoch - 1) * \
                self.data_loader.getTrainLen() + batch_idx
            self.writer.set_step(step_count)
            loss_res = self.train_loss.result()
            acc_res = self.train_accuracy.result()
            self.writer.scalar('train/step_loss', loss_res, step_count)
            self.writer.scalar('train/step_accuracy', acc_res, step_count)
            tbar.set_description('Train loss: %.3f' % loss_res)
        loss_res = self.train_loss.result()
        acc_res = self.train_accuracy.result()
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
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        tbar = tqdm(self.val_ds, total=self.data_loader.getValLen())
        for batch_idx, (imgs, frame_idx, target) in enumerate(tbar):
            predictions = self.model([imgs, frame_idx], training=False)
            loss = self.loss(target, predictions)

            self.test_loss(loss)
            self.test_accuracy(target, predictions)
            step_count = (epoch - 1) * self.data_loader.getValLen() + batch_idx
            self.writer.set_step(step_count, mode='valid')
            loss_res = self.test_loss.result()
            acc_res = self.test_accuracy.result()
            self.writer.scalar('val/step_loss', loss_res, step_count)
            self.writer.scalar('val/step_accuracy', acc_res, step_count)
            tbar.set_description('Test loss: %.3f' % loss_res)
        loss_res = self.test_loss.result()
        acc_res = self.test_accuracy.result()
        self.writer.scalar('val/loss', loss_res, epoch)
        self.writer.scalar('val/accuracy', acc_res, epoch)
        return {
            'loss': loss_res,
            'accuracy': acc_res
        }

    def test(self):
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        tbar = tqdm(self.val_ds, total=self.data_loader.getValLen())
        logit = []
        gTruth = []
        for batch_idx, (imgs, frame_idx, target) in enumerate(tbar):
            predictions = self.model([imgs, frame_idx], training=False)
            loss = self.loss(target, predictions)
            gTruth.append(target)
            logit.append(predictions)
            self.test_loss(loss)
            self.test_accuracy(target, predictions)
            tbar.set_description('Loss: %.3f' % self.test_loss.result())
        
        loss_res = self.test_loss.result()
        acc_res = self.test_accuracy.result()
        return {
            'loss': loss_res,
            'accuracy': acc_res
        }, logit, gTruth