from transformers import DistilBertConfig, RobertaConfig, TFRobertaModel
from .modeling_tf_distilbert import TFDistilBertModel



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


class VTNDistilBertLayer(TFDistilBertModel):
    def __init__(self,
                 embed_dim=768,
                 max_position_embeddings=200,
                 num_attention_heads=12,
                 num_hidden_layers=3,
                 pad_token_id=-1,
                 intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 hidden_dropout_prob=0.1):
        self.config = DistilBertConfig()
        self.config.type_vocab_size = 2
        self.config.hidden_dim = intermediate_size
        self.config.attention_dropout = attention_probs_dropout_prob
        self.config.dropout = hidden_dropout_prob
        self.config.n_layers = num_hidden_layers
        self.config.n_heads = num_attention_heads
        self.config.pad_token_id = pad_token_id
        self.config.max_position_embeddings = max_position_embeddings
        self.config.dim = embed_dim
        super(VTNDistilBertLayer, self).__init__(self.config)