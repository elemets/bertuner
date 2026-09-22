import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import Whitespace
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast


@pytest.fixture
def tiny_model_path(tmp_path):
    """A complete local checkpoint, requiring no Hub access."""
    path = tmp_path / "tiny-bert"
    vocab = {word: i for i, word in enumerate(
        ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "sample", "text"]
    )}
    backend = Tokenizer(WordPiece(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
        model_max_length=32,
    )
    tokenizer.save_pretrained(path)
    BertForSequenceClassification(BertConfig(
        vocab_size=len(vocab), hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
        num_labels=3,
    )).save_pretrained(path)
    return str(path)
