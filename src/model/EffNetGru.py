import tensorflow as tf


class EffNetBiGru(tf.keras.Model):
    def __init__(self, input_shape=[3, 15, 224, 224], backbone='EfficientNetB0', HIDDEN_DROPOUT=0.2,
                 MLP_DIM=768, n_classes=16, MLP_DROPOUT_RATE=0.4, RNN_UNIT=768, **kwargs):
        super(EffNetBiGru, self).__init__(**kwargs)
        self.img_input = tf.keras.layers.Input(input_shape)
        self.pos_input = tf.keras.layers.Input(input_shape[1])
        self.backbone = getattr(tf.keras.applications, backbone)(
            include_top=False, weights='imagenet', classes=n_classes,
            input_shape=[input_shape[2], input_shape[3], input_shape[0]])
        self.pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
        self.model_rest = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(RNN_UNIT, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(RNN_UNIT/2))),
            tf.keras.layers.Dropout(HIDDEN_DROPOUT),
            tf.keras.layers.Dense(MLP_DIM, activation='relu'),
            tf.keras.layers.Dropout(MLP_DROPOUT_RATE, name='top_dropout'),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])

        # self.out = self.call([self.img_input, self.pos_input])

        # super(EffNetBiGru, self).__init__(
        #     inputs=[self.img_input, self.pos_input], outputs=self.out, **kwargs)

    def call(self, x, training=False):
        x, position_ids = x
        shape = tf.shape(x)
        B, C, F, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x = tf.transpose(x, perm=[0, 2, 3, 4, 1])  # B, F, H, W, C
        x = tf.reshape(x, (B * F, H, W, C))
        x = self.backbone(x, training=training)
        x = self.pool(x)
        x = tf.reshape(x, (B, F, -1))
        x = self.model_rest(x)
        return x