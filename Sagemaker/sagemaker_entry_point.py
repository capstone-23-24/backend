import argparse
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from transformers import RobertaTokenizer, Trainer, TrainingArguments
from roberta_model import MyModel  # Import the MyModel class from roberta_model_class.py
from roberta_dataset import MyDataset  # Getting the MyDataset class from roberta_dataset.py

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    
    parser.add_argument('--train', type=str, default='/opt/ml/code/training_mini_data.csv')
    parser.add_argument('--test', type=str, default='/opt/ml/code/test_mini_data.csv')
    parser.add_argument('--output-dir', type=str, default='s3://capstone-19283/output/')
    parser.add_argument('--num-labels', type=int, default=7)
    args, _ = parser.parse_known_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data = pd.read_csv(args.train)
    test_data = pd.read_csv(args.test)

    # Initialize and configure your PyTorch model
    model = MyModel(num_labels=args.num_labels).to(device)
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    label_encoder = LabelEncoder()

    train_labels = train_data[['isP_bin', '1RE_val', '2RE_val', 'G_val', 'A_val']]
    encoded_train_labels = train_labels.apply(label_encoder.fit_transform)

    test_labels = test_data[['isP_bin', '1RE_val', '2RE_val', 'G_val', 'A_val']]
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

    # Save the trained model artifacts
    model.save_model(args.output_dir)

if __name__ == '__main__':
    main()
