import torch

from signature_verification.losses import ContrastiveLoss
from signature_verification.models import SiameseNetwork


def test_embedding_shape_and_norm():
    model = SiameseNetwork("resnet18", embedding_dim=32, pretrained=False).eval()
    with torch.no_grad():
        embedding = model.encode(torch.rand(2, 3, 64, 64))
    assert embedding.shape == (2, 32)
    assert torch.allclose(embedding.norm(dim=1), torch.ones(2), atol=1e-5)


def test_contrastive_loss_prefers_identical_positive_embeddings():
    loss = ContrastiveLoss(margin=0.5)
    same = torch.tensor([[1.0, 0.0]])
    different = torch.tensor([[0.0, 1.0]])
    good = loss(same, same, torch.ones(1))
    bad = loss(same, different, torch.ones(1))
    assert good < bad
