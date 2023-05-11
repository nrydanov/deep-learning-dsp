import logging
import os
from time import time

import torch
from models import get_model
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils import ParserType, init_device, init_logger, init_parser, save_history

def main():
    parser = init_parser(ParserType.TRAIN)
    args = parser.parse_args()
    init_logger(args)
    device = init_device(args.device)

    model = get_model(args.model_type)
    model_config = model.Settings(_env_file=args.model_config)
    model = model(model_config)
    model.to(device)

    optimizer = Adam(model.parameters(), args.learning_rate)

    save_path = f"checkpoints/{args.attempt_name}.pt"
    if args.restore_state is not None and args.restore_state:
        logging.info("Loading state from checkpoint")
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        last_epoch = checkpoint["last_epoch"]
        best_loss = checkpoint["best_loss"]
        logging.info("Successfully loaded state from checkpoint")
    elif os.path.exists(save_path):
        logging.error(
            "Attempt with such name is already exists, choose a different one"
        )
        return
    else:
        last_epoch = -1
        best_loss = 1e18

    provider = model.get_provider()
    data_config = provider.Settings(args.data_config)
    logging.info(f"Generating {data_config.total_samples} samples based on input")
    provider = provider(data_config)

    torch.manual_seed(69)
    train_provider, val_provider = random_split(provider, [0.8, 0.2])

    train_loader = DataLoader(train_provider, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_provider, batch_size=args.batch_size, shuffle=False)

    loss = MSELoss()
    n_val = len(train_loader)
    logging.info("Starting training loop")
    for epoch in range(last_epoch + 1, args.epochs):
        model.train()

        total_loss = 0
        loop = tqdm(train_loader)
        start_time = time()
        for i, (inputs, targets) in enumerate(loop):
            optimizer.zero_grad()

            targets = targets.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)

            train_loss = loss(outputs, targets)
            train_loss.backward()

            total_loss += train_loss.item()

            optimizer.step()
            loop.set_description(f"Epoch {epoch}/{args.epochs}")
            loop.set_postfix(loss=total_loss / (i + 1))
        train_time = time() - start_time
        train_loss = total_loss / len(train_loader)

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                targets = targets.to(device)
                inputs = inputs.to(device)

                outputs = model(inputs)
                val_loss = loss(outputs, targets)
                total_loss += val_loss.item()
                
        val_loss = total_loss / n_val

        save_path = f"checkpoints/{args.attempt_name}.pt"
        if val_loss < best_loss:
            print(f"\nValidation loss decreased from {best_loss:.4f} to {val_loss:.4f}")
            os.makedirs("checkpoints", exist_ok=True)
            best_loss = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "last_epoch": epoch,
                },
                save_path,
            )
        else:
            print(f"\nValidation loss didn't decrease, best loss: {best_loss:.4f}")
            checkpoint = torch.load(save_path)
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_model": checkpoint["best_model"],
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                    "last_epoch": epoch,
                },
                save_path,
            )
        history = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch": epoch,
            "train_time": train_time,
        }

        save_history(args.attempt_name, history)
        loop.set_postfix(val_loss=val_loss)


if __name__ == "__main__":
    main()
