import tensorflow as tf
from transformers import DistilBertConfig, RobertaConfig, TFRobertaModel, LongformerConfig
from .modeling_tf_longformer import TFLongformerModel
from .modeling_tf_distilbert import TFDistilBertModel


class VTNLongformerLayer(TFLongformerModel):
    def __init__(self,
                 embed_dim=768,
                 max_position_embeddings=2 * 60 * 60,
                 num_attention_heads=12,
                 num_hidden_layers=3,
                 attention_mode='sliding_chunks',
                 pad_token_id=-1,
                 attention_window=None,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1):
        self.config = LongformerConfig()
        self.config.attention_mode = attention_mode
        self.config.intermediate_size = intermediate_size
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.attention_dilation = [1, ] * num_hidden_layers
        self.config.attention_window = [
            256, ] * num_hidden_layers if attention_window is None else attention_window
        self.config.num_hidden_layers = num_hidden_layers
        self.config.num_attention_heads = num_attention_heads
        self.config.pad_token_id = pad_token_id
        self.config.max_position_embeddings = max_position_embeddings
        self.config.hidden_size = embed_dim
        super(VTNLongformerLayer, self).__init__(self.config)


def pad_to_window_size_local(
        input_ids,
        attention_mask,
        position_ids,
        one_sided_window_size: int,
        pad_token_id: int):
    w = 2 * one_sided_window_size
    seqlen = tf.shape(input_ids)[1]
    padding_len = (w - seqlen % w) % w
    input_ids = tf.pad(input_ids, tf.convert_to_tensor(
        [[0, 0], [0, padding_len], [0, 0]]), constant_values=pad_token_id)
    attention_mask = tf.pad(attention_mask, tf.convert_to_tensor(
        [[0, 0], [0, padding_len]]), constant_values=False)
    position_ids = tf.pad(position_ids, tf.convert_to_tensor(
        [[0, 0], [1, padding_len]]), constant_values=False)
    return input_ids, attention_mask, position_ids


class VTNRobertaLayer(TFRobertaModel):
    def __init__(self,
                 embed_dim=768,
                 max_position_embeddings=200,
                 num_attention_heads=12,
                 num_hidden_layers=3,
                 pad_token_id=-1,
                 attention_window=None,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1):
        self.config = RobertaConfig()
        self.config.intermediate_size = intermediate_size
        self.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.config.hidden_dropout_prob = hidden_dropout_prob
        self.config.attention_window = [
            256, ] * num_hidden_layers if attention_window is None else attention_window
        self.config.num_hidden_layers = num_hidden_layers
        self.config.num_attention_heads = num_attention_heads
        self.config.pad_token_id = pad_token_id
        self.config.max_position_embeddings = max_position_embeddings
        self.config.hidden_size = embed_dim
        super(VTNRobertaLayer, self).__init__(self.config)


class VTNDistilBertLayer(TFDistilBertModel):
    def __init__(self,
                 embed_dim=768,
                 max_position_embeddings=200,
                 num_attention_heads=12,
                 num_hidden_layers=3,
                 pad_token_id=-1,
                 attention_window=None,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1):
        self.config = DistilBertConfig()
        self.config.type_vocab_size = 2
        self.config.hidden_dim = intermediate_size
        self.config.attention_dropout = attention_probs_dropout_prob
        self.config.attention_window = [
            256, ] * num_hidden_layers if attention_window is None else attention_window
        self.config.dropout = hidden_dropout_prob
        self.config.n_layers = num_hidden_layers
        self.config.n_heads = num_attention_heads
        self.config.pad_token_id = pad_token_id
        self.config.max_position_embeddings = max_position_embeddings
        self.config.dim = embed_dim
        super(VTNDistilBertLayer, self).__init__(self.config)
