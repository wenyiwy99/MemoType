"""
File: router_training.py
Description: This file trains a basic BERT model as a router to classify memories.
"""
import os 
import sys
import torch
import argparse
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
MEM_TYPES = ['Episodic Memory', 'Personal Semantic Memory', 'General Semantic Memory']

def load_data(file_path):
    df = pd.read_csv(file_path)
    texts = df["Content"].tolist()

    multi_labels = []
    for i in range(len(df)):
        multi_label = []
        for category in MEM_TYPES:
            label = df.loc[i, category]
            if int(label) == 1: multi_label.append(1)
            else: multi_label.append(0)
        multi_labels.append(multi_label)
    return texts, multi_labels

class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.float),
        }

def train_epoch(model, data_loader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)

def main(num_labels, epochs, batch_size, learning_rate):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_path = os.path.join(base_dir, 'TriMEM/TriMEM.csv')
    save_path = os.path.join(base_dir, 'model/router/')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    texts, labels = load_data(load_path)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MultiLabelDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=num_labels, problem_type="multi_label_classification"
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    print("Starting training...")
    best_loss = float('inf') 
    for epoch in range(epochs):
        train_loss = train_epoch(model, loader, optimizer, device)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
            best_model_state = model.state_dict()

    if best_model_state is not None:
        print("Saving best model...")
        model.load_state_dict(best_model_state)  
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="training a basic BERT model as a router to classify memories")
    parser.add_argument("--num_labels", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    args = parser.parse_args()
    print(args)

    main(args.num_labels, args.epochs, args.batch_size, args.learning_rate)