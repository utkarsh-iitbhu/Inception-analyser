import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import logging
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HateSpeechDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Convert string labels to numeric
        self.data['label_numeric'] = self.data['label_binary']
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        label = self.data.iloc[idx]['label_numeric']
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def plot_confusion_matrix(true_labels, predictions, save_path='results/confusion_matrix.png'):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def train_model(train_df, test_df, model_save_path='models/hate_speech_model'):
    # Initialize MLflow
    mlflow.start_run()
    
    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
    
    # Log model parameters
    mlflow.log_param("model_type", "roberta-base")
    mlflow.log_param("max_length", 128)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("learning_rate", 2e-5)
    
    # Prepare datasets
    train_dataset = HateSpeechDataset(train_df, tokenizer)
    test_dataset = HateSpeechDataset(test_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)
    
    # Training settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 15
    
    # Variable to track the best F1 score and the corresponding model
    best_f1 = 0
    best_model = None
    best_tokenizer = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Training
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
        
        avg_train_loss = train_loss / len(train_loader)
        logger.info(f'Average training loss: {avg_train_loss}')
        mlflow.log_metric(f"train_loss_epoch_{epoch}", avg_train_loss)
        
        # Evaluation
        model.eval()
        predictions = []
        true_labels = []
        eval_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                eval_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        report = classification_report(true_labels, predictions, target_names=['noHate', 'hate'], output_dict=True)
        
        # Log metrics
        mlflow.log_metric(f"eval_loss_epoch_{epoch}", eval_loss / len(test_loader))
        mlflow.log_metric(f"f1_score_epoch_{epoch}", report['macro avg']['f1-score'])
        
        # Save best model based on F1 score
        current_f1 = report['macro avg']['f1-score']
        if current_f1 > best_f1:
            logger.info(f"New best F1 score: {current_f1}, saving model...")
            best_f1 = current_f1
            
            # Save the best model and tokenizer
            os.makedirs(model_save_path, exist_ok=True)
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            best_model = model
            best_tokenizer = tokenizer
            
            # Save confusion matrix
            plot_confusion_matrix(true_labels, predictions)
            
            # Save detailed results
            results = {
                'metrics': report,
                'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
                'predictions': predictions,
                'true_labels': true_labels
            }
            
            # Save results as CSV
            results_df = pd.DataFrame({
                'true_label': true_labels,
                'predicted_label': predictions,
                'text': test_df['text']
            })
            results_df.to_csv('results/test_predictions.csv', index=False)
    
    mlflow.end_run()
    
    # Return the best model and tokenizer after training
    return best_model, best_tokenizer


if __name__ == "__main__":
    # Load processed data
    train_df = pd.read_csv('data/processed_train.csv')
    test_df = pd.read_csv('data/processed_test.csv')
    
    # Train model
    model, tokenizer = train_model(train_df, test_df)