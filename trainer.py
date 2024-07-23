import tqdm
import torch

from torch.utils.data import DataLoader

from dataloader import BrainMriDataLoader
from loss import Loss

class Trainer:

    def __init__(self,
                 model: torch.nn.parallel.DistributedDataParallel,
                 dataloader_splits: BrainMriDataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 loss_fn: Loss,
                 gpu_id: int):
        self.model = model.module
        self.dataloader_splits = dataloader_splits
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn

        self.device = gpu_id

    def train(self, dataloader: DataLoader, loss_weights: dict, epoch: int):
        dataloader.sampler.set_epoch(epoch)

        train_loss = torch.zeros(1).to(self.device)
        train_dice = torch.zeros(1).to(self.device)
        
        self.model.train()
        for batch in dataloader:
            x, y = batch["image"].to(self.device), batch["mask"].to(self.device)
            weights = batch["diagnosis"]*loss_weights[1] + (1.0 - batch["diagnosis"])*loss_weights[0]
            self.optimizer.zero_grad(set_to_none=True)

            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y, weights)
            
            loss.backward()
            self.optimizer.step()

            train_loss += loss
            train_dice += self.loss_fn.dice_coeff_fn(y_pred, y)
        self.scheduler.step()

        return train_loss / len(dataloader), train_dice / len(dataloader)

    def test(self,
             dataloader: DataLoader,
             loss_weights: dict,
             epoch: int):
        dataloader.sampler.set_epoch(epoch)

        test_loss = torch.zeros(1).to(self.device)
        test_dice = torch.zeros(1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch["image"].to(self.device), batch["mask"].to(self.device)
                weights = batch["diagnosis"]*loss_weights[1] + (1.0 - batch["diagnosis"])*loss_weights[0]

                y_pred = self.model(x)
                test_loss += self.loss_fn(y_pred, y, weights)
                test_dice += self.loss_fn.dice_coeff_fn(y_pred, y)
        
        return test_loss / len(dataloader), test_dice / len(dataloader)

    def run(self, epochs: int = 50, save_freq: int = 10):
        train_dataloader = self.dataloader_splits.get_train_dataloader()
        val_dataloader = self.dataloader_splits.get_val_dataloader()
        test_dataloader = self.dataloader_splits.get_test_dataloader()

        response_cdf = self.dataloader_splits.get_class_imbalance()
        
        train_losses = torch.tensor([], device=self.device)
        val_losses = torch.tensor([], device=self.device)

        train_dices = torch.tensor([], device=self.device)
        val_dices = torch.tensor([], device=self.device)

        pbar = tqdm.tqdm(range(1, epochs+1))
        for epoch in pbar:
            train_loss, train_dice = self.train(train_dataloader, response_cdf, epoch)
            train_losses = torch.cat([train_losses, train_loss])
            train_dices = torch.cat([train_dices, train_dice])
            
            val_loss, val_dice = self.test(val_dataloader, response_cdf, epoch)
            val_losses = torch.cat([val_losses, val_loss])
            val_dices = torch.cat([val_dices, val_dice])

            avg_loss: torch.Tensor = torch.stack([train_losses, val_losses])
            torch.save(avg_loss, f"loss_curves.pt")
            avg_dice_coeff: torch.Tensor = torch.stack([train_dices, val_dices])
            torch.save(avg_dice_coeff, f"dice_coeff_curves.pt")

            pbar.set_postfix({
                "Epoch": f"{epoch}/{epochs}",
                "Train Loss": round(train_loss.item(), 2),
                "Val Loss": round(val_loss.item(), 2),
                "Train Dice": round(train_dice.item(), 2),
                "Val Dice": round(val_dice.item(), 2)
            })

            if self.device == 0 and epoch % save_freq == 0:
                torch.save(self.model.state_dict(), f"model_{epoch}.pt")
        
        if self.device == 0:
            test_loss, test_dice = self.test(test_dataloader, response_cdf, epoch)
            print(f"Test Loss: {round(test_loss.item(), 2)},",
                  f"Test Dice: {round(test_dice.item(), 2)}")