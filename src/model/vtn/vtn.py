import tensorflow as tf
from transformers import RobertaConfig, TFRobertaModel


class VTNRobertaLayer(TFRobertaModel):
    def __init__(self,
                 embed_dim=768,
                 max_position_embeddings=200,
                 num_attention_heads=12,
                 num_hidden_layers=3,
                 pad_token_id=-1,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1):
        self.config = RobertaConfig()
        self.config.intermediate_size = intermediate_size
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.num_hidden_layers = num_hidden_layers
        self.config.num_attention_heads = num_attention_heads
        self.config.pad_token_id = pad_token_id
        self.config.max_position_embeddings = max_position_embeddings
        self.config.hidden_size = embed_dim
        super(VTNRobertaLayer, self).__init__(self.config)


class VTN(tf.keras.Model):
    def __init__(self, cfg):
        super(VTN, self).__init__()
        self.backbone = getattr(tf.keras.applications, cfg.backbone)(
            include_top=False, weights='imagenet', input_shape=cfg.input_shape)

        self.bb_to_encoder = tf.keras.Sequential([
            tf.keras.layers.Dropout(cfg.HIDDEN_DROPOUT_PROB),
            tf.keras.layers.Dense(units=cfg.HIDDEN_DIM, activation='gelu')
        ])

        self.cls_token = tf.Variable(tf.random_normal_initializer(
            mean=0., stddev=1.)(shape=[1, 1, cfg.HIDDEN_DIM]))

        self.temporal_encoder = VTNRobertaLayer(
            embed_dim=cfg.HIDDEN_DIM,
            max_position_embeddings=cfg.MAX_POSITION_EMBEDDINGS,
            num_attention_heads=cfg.NUM_ATTENTION_HEADS,
            num_hidden_layers=cfg.NUM_HIDDEN_LAYERS,
            pad_token_id=cfg.PAD_TOKEN_ID,
            intermediate_size=cfg.INTERMEDIATE_SIZE,
            attention_probs_dropout_prob=cfg.ATTENTION_PROBS_DROPOUT_PROB,
            hidden_dropout_prob=cfg.HIDDEN_DROPOUT_PROB)

        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dense(units=cfg.MLP_DIM, activation='gelu'),
            tf.keras.layers.Dropout(cfg.MLP_DROPOUT_RATE),
            tf.keras.layers.Dense(cfg.n_classes, activation='softmax'),
        ])

    def call(self, x):
        x, position_ids = x
        shape = tf.shape(x)
        B, C, F, H, W = shape[0], shape[1], shape[2], shape[3], shape[4]
        x = tf.transpose(x, perm=[0, 2, 3, 4, 1])  # B, F, H, W, C
        x = tf.reshape(x, (B * F, H, W, C))
        x = self.backbone(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = self.bb_to_encoder(x)
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
        token_type_ids = tf.cast(tf.transpose(tf.pad(tf.zeros(
            [tf.shape(x)[1]-1, B]), [[1, 0], [0, 0]], constant_values=1)), tf.int64)

        position_ids = tf.pad(position_ids, tf.convert_to_tensor(
            [[0, 0], [0, 1]]), constant_values=0)
        position_ids = tf.cast(position_ids, tf.int64)
        mask = tf.cast(tf.not_equal(attention_mask, 0), tf.int32)
        max_position_embeddings = self.temporal_encoder.config.max_position_embeddings
        position_ids = position_ids % (max_position_embeddings - 2)

        position_ids = tf.where(
            tf.cast(tf.transpose(tf.pad(tf.zeros([tf.shape(position_ids)[1]-1, B]), [[1, 0], [0, 0]],
                                        constant_values=True)), tf.bool), tf.convert_to_tensor([max_position_embeddings - 2],
                                                                                               dtype=tf.int64), position_ids)
        position_ids = tf.where(
            mask == 0, tf.convert_to_tensor([max_position_embeddings - 1], dtype=tf.int64), position_ids)

        x = self.temporal_encoder(input_ids=None,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids,
                                  position_ids=position_ids,
                                  inputs_embeds=x,
                                  output_attentions=None,
                                  output_hidden_states=None,
                                  return_dict=None)
        x = x["last_hidden_state"]
        x = self.mlp_head(x[:, 0])
        return x
