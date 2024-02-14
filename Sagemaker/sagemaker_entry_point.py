import argparse
import pandas as pd
import torch
import logging
import json
from transformers import RobertaTokenizerFast, Trainer, TrainingArguments
from roberta_model import MyModel  # Import the MyModel class from roberta_model_class.py
from roberta_dataset import MyDataset  # Getting the MyDataset class from roberta_dataset.py

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def preprocess_data(file_path, tokenizer, label_map):
    data = pd.read_csv(file_path)
    tokenized_texts = []
    aligned_labels = []
    
    for _, row in data.iterrows():
        text = row['Text'] 
        labels = json.loads(row['label'])

        logger.info(f"text: {text}")
        logger.info(f"labels: {labels}")
        
        encoding = tokenizer(text, add_special_tokens=True, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        tokenized_texts.append(encoding)
        
        numerical_labels = [label_map[label["labels"][0]] for label in labels]
        aligned_labels.append(numerical_labels)
    
    return tokenized_texts, aligned_labels

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
    
    parser.add_argument('--train', type=str, default='/opt/ml/code/NER_training_data.csv')
    parser.add_argument('--test', type=str, default='/opt/ml/code/NER_test_data.csv')
    parser.add_argument('--output-dir', type=str, default='s3://capstone-19283/output/')
    parser.add_argument('--num-labels', type=int, default=7)
    args, _ = parser.parse_known_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Label map
    label_map = {
        'O': 0,          # Outside of any named entity
        'Person': 1,     # Beginning of a name
        'Location': 2,      # Beginning of a location
        '-100': -100     # Special token used to ignore subtokens in loss calculation
    }

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    model = MyModel(num_labels=args.num_labels).to(device)

    # Getting the Data and preprocessing
    train_texts, train_labels = preprocess_data(args.train, tokenizer, label_map)
    test_texts, test_labels = preprocess_data(args.test, tokenizer, label_map)

    train_dataset = MyDataset(texts=train_texts, labels=train_labels, tokenizer=tokenizer)
    test_dataset = MyDataset(texts=test_texts, labels=test_labels, tokenizer=tokenizer)

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
