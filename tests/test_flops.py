from tnms.config import model_shape
from tnms.flops import count_non_embedding_params
from tnms.model import Model


def test_param_count_for_s1_matches_manual_list():
    model = Model(**model_shape("S1"))
    modules = [model.norm_f]
    for block in model.blocks:
        modules += [block.norm1, block.attn.qkv, block.attn.proj,
                    block.norm2, block.mlp.gate, block.mlp.up, block.mlp.down]
    manual = sum(p.numel() for m in modules for p in m.parameters())
    assert count_non_embedding_params(model) == manual

    # The same number by hand: d = 64, SwiGLU hidden = 168, 2 layers.
    d, h = 64, 168
    per_block = 2 * d + 4 * d * d + 3 * d * h
    assert manual == 2 * per_block + d


def test_embeddings_are_not_counted():
    model = Model(**model_shape("S1"))
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for m in (model.tok_emb, model.pos_emb, model.image_proj, model.audio_proj)
                    for p in m.parameters())
    assert count_non_embedding_params(model) == total - embedding


def test_late_fusion_with_four_fewer_decoder_layers_matches_n():
    # Two 2-layer encoders add four blocks, so S3 with 2 decoder layers has the same N as S3.
    shape = model_shape("S3")
    early = Model(**shape)
    late = Model(shape["d_model"], shape["n_layers"] - 4, shape["n_heads"], encoder_layers=2)
    assert count_non_embedding_params(late) == count_non_embedding_params(early)
