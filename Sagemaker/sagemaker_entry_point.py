import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AdamW
from roberta_model import MyModel  # Import the MyModel class from roberta_model_class.py
from roberta_dataset import MyDataset  # Getting the MyDataset class from roberta_dataset.py

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        input_ids = batch['case']['input_ids'].to(device)
        attention_mask = batch['case']['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['case']['input_ids'].to(device)
            attention_mask = batch['case']['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(test_loader)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True, help='s3://sagemaker-us-east-1-131750570751/training_data.csv')
    parser.add_argument('--test-data', type=str, required=True, help='s3://sagemaker-us-east-1-131750570751/test_data.csv')
    parser.add_argument('--output-dir', type=str, required=True, help='s3://sagemaker-us-east-1-131750570751/Output/')
    parser.add_argument('--num-labels', type=int, required=True, help='7')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess your data
    train_data = pd.read_csv(args.train_data)
    test_data = pd.read_csv(args.test_data)

    label_encoder = LabelEncoder()

    train_labels = train_data[['isP_bin', '1RE_val', '2RE_val', 'G_val', 'A_val']]
    encoded_train_labels = train_labels.apply(label_encoder.fit_transform)

    test_labels = test_data[['isP_bin', '1RE_val', '2RE_val', 'G_val', 'A_val']]
    encoded_test_labels = test_labels.apply(label_encoder.transform)

    # Create instances of MyDataset
    train_dataset = MyDataset(case=train_data['case'].tolist(), labels=encoded_train_labels)
    test_dataset = MyDataset(case=test_data['case'].tolist(), labels=encoded_test_labels)

    # Create DataLoader for training
    batch_size = 32  # Adjust as needed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize and configure your PyTorch model
    model = MyModel(num_labels=args.num_labels).to(device)

    # Define optimizer and loss function
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 5  # Adjust as needed
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}')

    # Save the trained model artifacts
    model.save_pretrained(args.output_dir)
    
    # Create DataLoader for testing
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Testing loop
    test_loss = test(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss}')

    # Save the trained model artifacts
    model.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
