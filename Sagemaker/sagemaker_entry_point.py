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
        input_ids_part1 = batch['case_part1']['input_ids'].to(device)
        attention_mask_part1 = batch['case_part1']['attention_mask'].to(device)
        input_ids_part2 = batch['case_part2']['input_ids'].to(device)
        attention_mask_part2 = batch['case_part2']['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids_part1=input_ids_part1, attention_mask_part1=attention_mask_part1,
                        input_ids_part2=input_ids_part2, attention_mask_part2=attention_mask_part2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data', type=str, required=True, help='S3 path to training data')
    parser.add_argument('--validation-data', type=str, required=True, help='S3 path to validation data')
    parser.add_argument('--output-dir', type=str, required=True, help='S3 path for saving model artifacts')
    parser.add_argument('--num-labels', type=int, required=True, help='Number of output labels')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess your data
    train_data = pd.read_csv(args.train_data)
    validation_data = pd.read_csv(args.validation_data)

    label_encoder = LabelEncoder()

    train_case_part1 = train_data['casePart1'].tolist()
    train_case_part2 = train_data['casePart2'].tolist()
    train_labels_str = train_data[['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10']]
    train_labels = label_encoder.fit_transform(train_labels_str.values.flatten()).reshape(train_labels_str.shape)

    validation_case_part1 = validation_data['casePart1'].tolist()
    validation_case_part2 = validation_data['casePart2'].tolist()
    validation_labels_str = validation_data[['label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8', 'label9', 'label10']]
    validation_labels = label_encoder.transform(validation_labels_str.values.flatten()).reshape(validation_labels_str.shape)

    # Create instances of MyDataset
    train_dataset = MyDataset(case_part1=train_case_part1, case_part2=train_case_part2, labels=train_labels)
    validation_dataset = MyDataset(case_part1=validation_case_part1, case_part2=validation_case_part2, labels=validation_labels)

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
