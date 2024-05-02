from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional
import wandb
from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, List, Union
from audioDeepFake.data import create_dataset
from .trainer import ModelTrainer
from pathlib import Path
import torchaudio 
import functools
import wandb 

import sys
sys.path.append("audioDeepFake")
from data import load_directory_split_train_test
from data import _build_preprocessing

from utils import alt_compute_eer, save_checkpoint

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mfcc = functools.partial(_build_preprocessing, transform=torchaudio.transforms.MFCC)

class Trainer(object):
    def __init__(
        self,
        epochs: int,
        batch_size: int,
        device: str,
        lr: float = 1e-3,
        optimizer_fn: Callable = torch.optim.Adam,
        optimizer_kwargs: Optional[dict] = {},
    ) -> None:
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.device = device
        self.lr = lr
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.optimizer_kwargs["lr"] = self.lr


class ModelTrainer(Trainer):
    def train(
        self,
        model_classname:str,
        model: nn.Module,
        dataset_train: Dataset,
        dataset_test: Dataset,  
        save_dir: Union[str, Path] = None, 
        run_name = "Training",
        project_name = "Example" 
    ) :
        if save_dir:
            save_dir: Path = Path(save_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

        train_loader = DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        test_loader = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            drop_last=False,
        )

        # Initialize wandb run
        wandb.init(project=project_name, name=run_name)

        criterion = nn.BCEWithLogitsLoss()
        optim = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        for param_group in optim.param_groups:
            param_group["lr"] = self.lr

        best_model = None
        best_acc = 0
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            num_correct = 0.0
            num_total = 0.0

            for _, (batch_x, _, _, batch_y) in enumerate(train_loader):
                if model_classname == "TDNN":
                    batch_x = batch_x.view(batch_x.size(0), -1)

                curr_batch_size = batch_x.size(0)
                num_total += curr_batch_size
                batch_x = batch_x.to(self.device)

                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                batch_out = model(batch_x)  
                
                batch_loss = criterion(batch_out, batch_y) 
                batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
                
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()
                total_loss += batch_loss.item() * curr_batch_size
                
                optim.zero_grad()
                batch_loss.backward()  
                optim.step() 

            total_loss /= num_total
            train_acc = (num_correct / num_total) * 100

            model.eval()
            num_correct = 0.0
            num_total = 0.0
            y_true = []
            y_pred = []

            for batch_x, _, _, batch_y in test_loader:
                if model_classname == "TDNN":
                    batch_x = batch_x.view(batch_x.size(0), -1)

                curr_batch_size = batch_x.size(0)
                num_total += curr_batch_size

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
                
                y_true.append(batch_y.clone().detach().int().cpu())
                batch_out = model(batch_x)
                batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
                
                y_pred.append(batch_pred.clone().detach().cpu())
                num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

            test_acc = (num_correct / num_total) * 100
            y_true: np.ndarray = torch.cat(y_true, dim=0).numpy()
            y_pred: np.ndarray = torch.cat(y_pred, dim=0).numpy()
            test_eer = alt_compute_eer(y_true, y_pred)

            print(
                f"[{epoch:03d}]: loss: {round(total_loss, 4)} - train acc: {round(train_acc, 2)} - test acc: {round(test_acc, 2)} - test eer : {round(test_eer, 4)}"
            )
            
            wandb.log({'total_loss': total_loss, 'train acc': train_acc , 'test acc': test_acc , 'test eer' : test_eer})

            if test_acc > best_acc:
                best_acc = test_acc

                if save_dir:
                    save_path = save_dir / "best.pt"
                    save_checkpoint(
                        epoch=epoch,
                        model=model,
                        optimizer=optim,
                        model_kwargs=self.__dict__,
                        filename=save_path,
                    )
        return None

    def eval(
        self,
        model_classname:str,
        model: nn.Module,
        dataset_test: Dataset,
        save_dir: Union[str, Path] = None,
    ) -> None:
        if save_dir:
            save_dir: Path = Path(save_dir)
            if not save_dir.exists():
                save_dir.mkdir(parents=True)

        test_loader = DataLoader(
            dataset_test,
            batch_size=self.batch_size,
            drop_last=False,
        )

        model.eval()
        num_correct = 0.0
        num_total = 0.0
        y_true = []
        y_pred = []

        for batch_x, _, _, batch_y in test_loader:
            if model_classname == "TDNN":
                    batch_x = batch_x.view(batch_x.size(0), -1)
            curr_batch_size = batch_x.size(0)
            num_total += curr_batch_size
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)
            y_true.append(batch_y.clone().detach().int().cpu())
            batch_out = model(batch_x)
            batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
            y_pred.append(batch_pred.clone().detach().cpu())
            num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()

        test_acc = (num_correct / num_total) * 100
        y_true: np.ndarray = torch.cat(y_true, dim=0).numpy()
        y_pred: np.ndarray = torch.cat(y_pred, dim=0).numpy()
        test_eer = alt_compute_eer(y_true, y_pred)

        print(f"test acc: {round(test_acc, 2)} - test eer : {round(test_eer, 4)}")
        
        wandb.log({'test_acc': test_acc, 'test_eer': test_eer})



def train(
    model,
    epochs: int = 20,
    device = DEVICE,
    batch_size: int = 32,
    save_dir: Union[str, Path] = None,
    model_classname: str = "ShallowCNN",
):
    feature_fn = eval("mfcc")
    
    real_dataset_train, real_dataset_test = load_directory_split_train_test(
        path=real_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=0.2,
        use_double_delta=True,
        pad=True,
        label=1,
    )

    fake_melgan_train, fake_melgan_test = load_directory_split_train_test(
        path=fake_dir,
        feature_fn=feature_fn,
        feature_kwargs={},
        test_size=0.2,
        use_double_delta=True,
        pad=True,
        label=0,
    )

    dataset_train, dataset_test = None, None
    dataset_train = ConcatDataset([real_dataset_train, fake_melgan_train])
    dataset_test = ConcatDataset([real_dataset_test, fake_melgan_test])

    ModelTrainer(
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        lr=0.0001,
        optimizer_kwargs={"weight_decay": 0.0001},
    ).train(
        model_classname = model_classname,
        model=model,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        save_dir=save_dir)


def experiment(
        name,
        epochs,
        batch_size,
        model,
        model_classname,
        evaluate_only=False,
        run_name=None, 
        project_name=None,
    **kwargs,
):
    root_save_dir = Path("saved")
    save_dir = root_save_dir / name
    save_dir.mkdir(parents=True, exist_ok=True)
        
    train(model,
        epochs=epochs,
        device=DEVICE,
        batch_size=batch_size,
        save_dir=save_dir,
        model_classname=model_classname,
        run_name=run_name,
        project_name=project_name
    )

