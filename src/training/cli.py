import argparse
from src.training.train import train_model

def main():
    parser = argparse.ArgumentParser(description="Hindi Handwriting AI - Milestone A CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model on the synthetic dataset")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    
    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume training from a checkpoint")
    resume_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint.pt")
    resume_parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs to reach")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model checkpoint")
    eval_parser.add_argument("--checkpoint", type=str, required=True)
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate handwriting from text")
    gen_parser.add_argument("--text", type=str, required=True)
    gen_parser.add_argument("--checkpoint", type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == "train":
        print("Starting training...")
        train_model(epochs=args.epochs)
    elif args.command == "resume":
        print(f"Resuming training from {args.checkpoint}...")
        train_model(epochs=args.epochs, resume_checkpoint=args.checkpoint)
    elif args.command == "evaluate":
        print("Evaluation pipeline not fully wired to CLI yet.")
    elif args.command == "generate":
        print(f"Generation for '{args.text}' using {args.checkpoint} not fully wired yet.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
