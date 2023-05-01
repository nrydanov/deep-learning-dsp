import torch
import os

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.nn import MSELoss, Module
from models import get_model
from utils import init_parser, init_device, empty_cache

from keras.utils import Progbar


def main():
    parser = init_parser()
    args = parser.parse_args()
    device: torch.device = init_device(args.device)

    model = get_model(args.model_type)

    config = model.Settings(_env_file=args.model_config)
    model: torch.Module = model(config)
    model.to(device)

    provider = model.get_provider()
    provider = provider(provider.Settings(args.data_config))

    torch.manual_seed(69)
    train_provider, val_provider = random_split(provider, [0.8, 0.2])

    train_loader = DataLoader(train_provider, batch_size=args.batch_size)
    val_loader = DataLoader(val_provider, batch_size=args.batch_size)

    optimizer = Adam(model.parameters(), args.learning_rate)

    if args.restore_state is not None and args.restore_state == True:
        checkpoint = torch.load(args.save_path)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        last_epoch = checkpoint["last_epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        last_epoch = -1
        best_loss = 1e18

    criteria = MSELoss()

    n_train = len(train_loader)
    n_val = len(val_loader)

    for epoch in range(last_epoch + 1, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        pb = Progbar(target=n_train)
        model.train()

        total_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()

            targets = targets.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)

            train_loss = criteria(outputs, targets)
            train_loss.backward()

            total_loss += train_loss.item()

            optimizer.step()
            pb.update(i, values=[("loss", total_loss / (i + 1))])

        model.eval()

        with torch.no_grad():
            total_loss = 0
            for i, (inputs, targets) in enumerate(val_loader):
                targets = targets.to(device)
                inputs = inputs.to(device)

                outputs = model(inputs)
                val_loss = criteria(outputs, targets)
                total_loss += val_loss.item()

            current_loss = total_loss / n_val
            if current_loss < best_loss:
                print(
                    f"\nValidation loss decreased from {best_loss:.4f} to {current_loss:.4f}"
                )
                dir_path = "/".join(args.save_path.split("/")[:-1])
                os.makedirs(dir_path, exist_ok=True)
                best_loss = total_loss / n_val
                torch.save(
                    {
                        "model": model.state_dict(),
                        "best_model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "last_epoch": epoch,
                    },
                    args.save_path,
                )
            else:
                checkpoint = torch.load(args.save_path)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "best_model": checkpoint["best_model"],
                        "optimizer": optimizer.state_dict(),
                        "best_loss": best_loss,
                        "last_epoch": epoch,
                    },
                    args.save_path,
                )

            pb.update(i, values=[("val_loss", total_loss / len(val_loader))])
            print("\r")


if __name__ == "__main__":
    main()
