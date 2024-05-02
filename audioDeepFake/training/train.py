import os
import torch
from torch.utils.data import ConcatDataset, DataLoader
from typing import Dict, List, Union
from audioDeepFake.data import create_dataset
from .trainer import experiment

import sys
sys.path.append("audioDeepFake")
from models import ShallowCNN, SimpleLSTM, DTDNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

real_dir = "/kaggle/input/the-lj-speech-dataset/LJSpeech-1.1"
fake_dir = "/kaggle/input/wavefake-test/generated_audio"

melgan_dir = os.path.join(fake_dir,"ljspeech_melgan")
test_size = 0.2

real_dataset_train, real_dataset_test = create_dataset(real_dir, label=1)
fake_melgan_train, fake_melgan_test = create_dataset(melgan_dir, label=0)

dataset_train = ConcatDataset([real_dataset_train, fake_melgan_train])
dataset_test = ConcatDataset([real_dataset_test, fake_melgan_test])

train_dataloader = DataLoader(dataset_train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=64, shuffle=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
epochs = 10
seed = 42


model = ShallowCNN(in_features= 1,out_dim=1).to(DEVICE)
model_classname = 'ShallowCNN'
exp_name = f"{model_classname}"

experiment(
    name=exp_name,
    epochs=epochs,
    batch_size=batch_size,
    model=model,
    model_classname=model_classname,
    evaluate_only=False,
    run_name="ShallowCNN Training", 
    project_name="AudioDeepFake"
)


model = SimpleLSTM(feat_dim= 40, time_dim= 972, mid_dim= 30, out_dim= 1).to(DEVICE)

model_classname = 'SimpleLSTM'
exp_name = f"{model_classname}"

experiment(
    name=exp_name,
    epochs=epochs,
    batch_size=batch_size,
    model = model,
    model_classname=model_classname,
    evaluate_only=False,
    run_name="SimpleLSTM Training", 
    project_name="AudioDeepFake"
)


model_classname = 'TDNN'
model = DTDNN(feat_dim= 38880, num_classes=1).to(DEVICE)
exp_name = f"{model_classname}"

experiment(
    name=exp_name,
    epochs=epochs,
    batch_size=batch_size,
    model = model,
    model_classname=model_classname,
    evaluate_only=False,
    run_name="DTDNN Training", 
    project_name="AudioDeepFake"
)
