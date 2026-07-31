import os
import torch
import pytest
import shutil
from src.config.production import ProductionConfig
from src.models.production.model import ProductionHandwritingModel
from src.training.trainer import ProductionTrainer

@pytest.fixture
def config():
    cfg = ProductionConfig()
    cfg.architecture.text_embedding_dim = 16
    cfg.architecture.text_encoder_hidden_dim = 32
    cfg.architecture.decoder_hidden_dim = 64
    cfg.architecture.mdn_mixtures = 5
    cfg.training.epochs = 1
    cfg.training.mixed_precision = False
    cfg.checkpoint_dir = "test_checkpoints/"
    cfg.log_dir = "test_logs/"
    return cfg

@pytest.fixture
def model(config):
    return ProductionHandwritingModel(config=config, vocab_size=50)

def test_production_model_forward(model):
    """Test the forward pass and shape constraints."""
    B, U, S = 4, 10, 20
    text_tokens = torch.randint(0, 50, (B, U))
    text_lengths = torch.tensor([10, 8, 5, 2])
    coords = torch.randn(B, S, 3) # Batch, Seq, Feats
    
    mdn_params, attention_weights = model(text_tokens, text_lengths, coords)
    pi, mu1, mu2, sigma1, sigma2, rho, eos = mdn_params
    
    assert pi.shape == (B, S, 5) # 5 mixtures
    assert mu1.shape == (B, S, 5)
    assert sigma1.shape == (B, S, 5)
    assert rho.shape == (B, S, 5)
    assert eos.shape == (B, S, 1)
    
    assert attention_weights.shape == (B, S, U)
    
    # Sigmas must be > 0
    assert torch.all(sigma1 > 0)
    assert torch.all(sigma2 > 0)
    
    # Rho must be in (-1, 1)
    assert torch.all((rho > -1.0) & (rho < 1.0))

def test_production_trainer_gradient_stability(model, config):
    """Test that the backward pass flows without NaNs."""
    trainer = ProductionTrainer(model, config, device="cpu")
    
    # Dummy data
    B, U, S = 2, 5, 10
    text_tokens = torch.randint(0, 50, (B, U))
    text_lengths = torch.tensor([5, 5])
    coords = torch.randn(B, S+1, 3) # S+1 for input-target shifted by 1
    # Pen state must be in [0, 1] for BCE loss
    coords[..., 2] = torch.rand(B, S+1)
    
    # Create simple dataloader
    dataset = torch.utils.data.TensorDataset(text_tokens, text_lengths, coords)
    loader = torch.utils.data.DataLoader(dataset, batch_size=B)
    
    loss = trainer.train_epoch(loader)
    
    assert loss > 0
    assert not torch.isnan(torch.tensor(loss))

def test_checkpoint_save_and_load(model, config):
    """Test if checkpoint perfectly restores states."""
    trainer = ProductionTrainer(model, config, device="cpu")
    trainer.global_step = 100
    trainer.best_loss = 0.5
    trainer.current_epoch = 5
    
    trainer.save_checkpoint(os.path.join(config.checkpoint_dir, "test.pt"))
    
    # New trainer
    model_new = ProductionHandwritingModel(config=config, vocab_size=50)
    trainer_new = ProductionTrainer(model_new, config, device="cpu")
    
    trainer_new.load_checkpoint(os.path.join(config.checkpoint_dir, "test.pt"))
    
    assert trainer_new.global_step == 100
    assert trainer_new.best_loss == 0.5
    assert trainer_new.current_epoch == 5
    
    # Cleanup
    shutil.rmtree(config.checkpoint_dir)
    shutil.rmtree(config.log_dir)
