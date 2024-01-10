import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaForSequenceClassification, RobertaTokenizer, AdamW
from roberta_model_class import MyModel  # Import the MyModel class from roberta_model_class.py
from roberta_dataset_class import MyDataset # Getting the myDataset class from roberta_dataset.py

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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True, help='S3 path to training data')
    parser.add_argument('--test-data', type=str, required=True, help='S3 path to validation data')
    parser.add_argument('--output-dir', type=str, required=True, help='S3 path for saving model artifacts')
    parser.add_argument('--num-labels', type=int, required=True, help='Number of output labels')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess your data
    train_data = pd.read_csv(args.train_data)
    test_data = pd.read_csv(args.test_data)

    label_encoder = LabelEncoder()

    train_case = train_data['case'].tolist()
    train_labels_str = train_data[['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10']]
    train_labels = label_encoder.fit_transform(train_labels_str.values.flatten()).reshape(train_labels_str.shape)

    test_case = test_data['case'].tolist()
    test_labels_str = test_data[['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10']]
    test_labels = label_encoder.transform(test_labels_str.values.flatten()).reshape(test_labels_str.shape)

    # Create instances of MyDataset
    train_dataset = MyDataset(case=train_case, labels=train_labels)
    test_dataset = MyDataset(case_=test_case, labels=test_labels)

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

if __name__ == '__main__':
    main()
