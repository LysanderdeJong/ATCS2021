from pl_model import SentenceEmbeddings


def get_model(TEXT, args):
    dict_args = vars(args)
    return SentenceEmbeddings(TEXT.vocab.vectors, **dict_args)