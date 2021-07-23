from utils import initializer


class VTNOptions:
    @initializer
    def __init__(self, backbone="EfficientNetB0", n_classes=16, MLP_DIM=384, input_shape=[3, 16, 224, 224],
                 HIDDEN_DIM=384, MAX_POSITION_EMBEDDINGS=288, NUM_ATTENTION_HEADS=12, NUM_HIDDEN_LAYERS=3,
                 PAD_TOKEN_ID=-1, INTERMEDIATE_SIZE=2048, ATTENTION_PROBS_DROPOUT_PROB=0.1, ATTENTION_WINDOW=[18, 18, 18],
                 HIDDEN_DROPOUT_PROB=0.1, MLP_DROPOUT_RATE=0.4, temporal="VTNRobertaLayer"):
        self.input_shape = tuple(input_shape)


if __name__ == '__main__':
    opt = VTNOptions()
    print(opt.backbone)
