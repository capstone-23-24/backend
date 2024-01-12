import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from transformers import AdamW, RobertaTokenizer, Trainer, TrainingArguments
from roberta_model import MyModel  # Import the MyModel class from roberta_model_class.py
from roberta_dataset import MyDataset  # Getting the MyDataset class from roberta_dataset.py

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    
    # parser.add_argument('--train-data', type=str, default='s3://sagemaker-us-east-1-131750570751/training_data.csv')
    # parser.add_argument('--test-data', type=str, default='s3://sagemaker-us-east-1-131750570751/test_data.csv')
    parser.add_argument('--train', type=str, default='./training_data.csv')
    parser.add_argument('--test', type=str, default='./test_data.csv')
    parser.add_argument('--output-dir', type=str, default='s3://sagemaker-us-east-1-131750570751/Output/')
    parser.add_argument('--num-labels', type=int, default=7)
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = pd.read_csv(args.train)
    test_data = pd.read_csv(args.test)

    # # Load and preprocess your data
    # train_data = pd.read_csv(args.train_data)
    # test_data = pd.read_csv(args.test_data)

    # Initialize and configure your PyTorch model
    model = MyModel(num_labels=args.num_labels).to(device)
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    label_encoder = LabelEncoder()

    train_labels = train_data[['Text', 'isP_bin', '1RE_val', '2RE_val', 'G_val', 'A_val']]
    encoded_train_labels = train_labels.apply(label_encoder.fit_transform)

    test_labels = test_data[['Text', 'isP_bin', '1RE_val', '2RE_val', 'G_val', 'A_val']]
    encoded_test_labels = test_labels.apply(label_encoder.fit_transform)

    # Create instances of MyDataset
    train_dataset = MyDataset(case=train_data['Text'].tolist(), labels=encoded_train_labels, tokenizer=tokenizer)
    test_dataset = MyDataset(case=test_data['Text'].tolist(), labels=encoded_test_labels, tokenizer=tokenizer)
    
    # define training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        evaluation_strategy="epoch",
        logging_dir=f"{args.output_dir}/logs",
        learning_rate=float(args.learning_rate),
    )

    # create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
#         compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    # train model
    trainer.train()
    
    # label_encoder = LabelEncoder()

    # train_labels = train_data[['Text', 'isP_bin', '1RE_val', '2RE_val', 'G_val', 'A_val']]
    # encoded_train_labels = train_labels.apply(label_encoder.fit_transform)

    # test_labels = test_data[['Text', 'isP_bin', '1RE_val', '2RE_val', 'G_val', 'A_val']]
    # encoded_test_labels = test_labels.apply(label_encoder.fit_transform)


    # # Create instances of MyDataset
    # train_dataset = MyDataset(case=train_data['Text'].tolist(), labels=encoded_train_labels)
    # test_dataset = MyDataset(case=test_data['Text'].tolist(), labels=encoded_test_labels)
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # # Create instances of MyDataset
    # train_dataset = MyDataset(case=train_data['Text'].tolist(), labels=encoded_train_labels, tokenizer=tokenizer)
    # test_dataset = MyDataset(case=test_data['Text'].tolist(), labels=encoded_test_labels, tokenizer=tokenizer)

    # # Create DataLoader for training
    # batch_size = 32  # Adjust as needed
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    # # Define optimizer and loss function
    # optimizer = AdamW(model.parameters(), lr=1e-5)
    # criterion = nn.CrossEntropyLoss()

    # # Training loop
    # num_epochs = 5  # Adjust as needed
    # for epoch in range(num_epochs):
    #     train_loss = train(model, train_loader, optimizer, criterion, device)
    #     print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}')

    # # Save the trained model artifacts
    # model.save_pretrained(args.output_dir)
    
    # # Create DataLoader for testing
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # Testing loop
    # test_loss = test(model, test_loader, criterion, device)
    # print(f'Test Loss: {test_loss}')

    # # Save the trained model artifacts
    # model.save_pretrained(args.output_dir)

if __name__ == '__main__':
    main()
