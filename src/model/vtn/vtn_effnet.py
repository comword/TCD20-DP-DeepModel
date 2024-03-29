import tensorflow as tf
from . import layers
from .. import efficientnet

class VTN(tf.keras.Model):
    def __init__(self, cfg, **kwargs):
        super(VTN, self).__init__(**kwargs)

        self.img_input = tf.keras.layers.Input(cfg.input_shape)
        self.pos_input = tf.keras.layers.Input(cfg.input_shape[1])
        self.backbone = getattr(efficientnet, cfg.backbone)(
            include_top=False, weights='imagenet', top_conv_dim=cfg.HIDDEN_DIM,
            input_shape=[cfg.input_shape[2], cfg.input_shape[3], cfg.input_shape[0]])

        self.bb_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.bb_pooling = tf.keras.layers.GlobalMaxPooling2D()

        self.cls_token = tf.Variable(tf.random_normal_initializer(
            mean=0., stddev=1.)(shape=[1, 1, cfg.HIDDEN_DIM]))

        self.temporal_encoder = getattr(layers, cfg.temporal)(
            embed_dim=cfg.HIDDEN_DIM,
            max_position_embeddings=cfg.MAX_POSITION_EMBEDDINGS,
            num_attention_heads=cfg.NUM_ATTENTION_HEADS,
            num_hidden_layers=cfg.NUM_HIDDEN_LAYERS,
            pad_token_id=cfg.PAD_TOKEN_ID,
            attention_window=cfg.ATTENTION_WINDOW,
            intermediate_size=cfg.INTERMEDIATE_SIZE,
            attention_probs_dropout_prob=cfg.ATTENTION_PROBS_DROPOUT_PROB,
            hidden_dropout_prob=cfg.HIDDEN_DROPOUT_PROB)

        self.mlp_norm = tf.keras.layers.LayerNormalization()
        self.mlp_dense = tf.keras.layers.Dense(
            units=cfg.MLP_DIM, activation='gelu')
        self.mlp_dropout = tf.keras.layers.Dropout(cfg.MLP_DROPOUT_RATE)
        self.mlp_output = tf.keras.layers.Dense(
            cfg.n_classes, activation='softmax')

        self.out = self.call([self.img_input, self.pos_input])

        super(VTN, self).__init__(
            inputs=[self.img_input, self.pos_input], outputs=self.out, **kwargs)

    def call(self, x, training=False):
        x, position_ids = x
        shape = tf.shape(x)
        B, C, F, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x = tf.transpose(x, perm=[0, 2, 3, 4, 1])  # B, F, H, W, C
        x = tf.reshape(x, (B * F, H, W, C))
        x = self.backbone(x, training=training)
        x = self.bb_norm(x)
        x = self.bb_pooling(x)
        x = tf.reshape(x, (B, F, -1))

        shape = tf.shape(x)
        B, D, E = shape[0], shape[1], shape[2]
        attention_mask = tf.ones([B, D])
        cls_tokens = tf.broadcast_to(self.cls_token, [B, 1, E])
        x = tf.concat([cls_tokens, x], 1)
        cls_atten = tf.broadcast_to(tf.ones(1), [B, 1])
        attention_mask = tf.concat([attention_mask, cls_atten], 1)
        attention_mask = tf.where(
            tf.cast(tf.transpose(tf.pad(tf.zeros([D, B]), [[1, 0], [0, 0]], constant_values=True)),
                    tf.bool), 2., attention_mask)
        x, attention_mask, position_ids = layers.pad_to_window_size_local(
            x,
            attention_mask,
            position_ids,
            self.temporal_encoder.config.attention_window[0],
            self.temporal_encoder.config.pad_token_id)
        token_type_ids = tf.cast(tf.transpose(tf.pad(tf.zeros(
            [tf.shape(x)[1]-1, B]), [[1, 0], [0, 0]], constant_values=1)), tf.int32)

        # position_ids = tf.pad(position_ids, tf.convert_to_tensor(
        #     [[0, 0], [1, 0]]), constant_values=0)
        position_ids = tf.cast(position_ids, tf.int32)
        mask = tf.cast(tf.not_equal(attention_mask, 0), tf.int32)
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)

        position_ids = tf.where(
            tf.cast(tf.transpose(tf.pad(tf.zeros([tf.shape(position_ids)[1]-1, B]), [[1, 0], [0, 0]],
                                        constant_values=True)), tf.bool), tf.convert_to_tensor([max_position_embeddings - 2],
                                                                                               dtype=tf.int32), position_ids)
        position_ids = tf.where(
            mask == 0, tf.convert_to_tensor([max_position_embeddings - 1], dtype=tf.int32), position_ids)

        x = self.temporal_encoder(input_ids=None,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=x,
                                  output_attentions=None,
                                  output_hidden_states=None,
                                  return_dict=None,
                                  training=training)
        x = x["last_hidden_state"]
        x = self.mlp_norm(x[:, 0])
        x = self.mlp_dense(x)
        x = self.mlp_dropout(x, training=training)
        x = self.mlp_output(x)
        return x
