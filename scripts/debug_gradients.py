import torch
from src.config.production import ProductionConfig
from src.models.production.model import ProductionHandwritingModel
from src.training.trainer import ProductionTrainer

def debug_gradients():
    config = ProductionConfig()
    config.architecture.text_embedding_dim = 16
    config.architecture.text_encoder_hidden_dim = 32
    config.architecture.decoder_hidden_dim = 64
    config.architecture.mdn_mixtures = 5
    config.training.mixed_precision = False
    
    model = ProductionHandwritingModel(config=config, vocab_size=50)
    trainer = ProductionTrainer(model, config, device="cpu")
    
    B, U, S = 2, 5, 10
    text_tokens = torch.randint(0, 50, (B, U))
    text_lengths = torch.tensor([5, 5])
    coords = torch.randn(B, S+1, 3) 
    coords[..., 2] = torch.rand(B, S+1)
    
    dataset = torch.utils.data.TensorDataset(text_tokens, text_lengths, coords)
    loader = torch.utils.data.DataLoader(dataset, batch_size=B)
    
    loss = trainer.train_epoch(loader)
    
    print(f"Loss: {loss}")
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"NONE GRADIENT: {name}")
        else:
            print(f"HAS GRADIENT: {name} (Sum: {param.grad.abs().sum().item():.4f})")
            
if __name__ == "__main__":
    debug_gradients()
