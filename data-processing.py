import pandas as pd
import os
import logging
import requests
from zipfile import ZipFile
from glob import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_URL = 'https://github.com/Vicomtech/hate-speech-dataset/archive/refs/heads/master.zip'
EXTRACT_DIR = 'data'

def download_and_extract_data(url, extract_to):
    """
    Download and extract data from the provided GitHub URL.
    
    Args:
        url: URL of the zip file to download.
        extract_to: Directory where the zip contents will be extracted.
    """
    # Create directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    zip_file_path = os.path.join(extract_to, 'hate_speech_dataset.zip')
    
    # Download the dataset
    logger.info("Downloading dataset...")
    response = requests.get(url)
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the dataset
    logger.info("Extracting dataset...")
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Remove the zip file after extraction
    os.remove(zip_file_path)
    logger.info("Download and extraction completed.")

def read_tweet_file(file_path):
    """
    Read content from a single tweet file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        return None

def process_dataset(data_folder, annotations_file):
    """
    Process the dataset by combining tweet files with their annotations.
    
    Args:
        data_folder: Path to folder containing tweet text files.
        annotations_file: Path to annotation_metadata.csv.
    """
    logger.info(f"Processing data from folder: {data_folder}")
    
    # Read annotations
    annotations_df = pd.read_csv(annotations_file)
    
    # Clean up the annotations DataFrame
    annotations_df['file_id'] = annotations_df['file_id'].str.split('_').str[0] + '_' + \
                               annotations_df['file_id'].str.split('_').str[1]
    
    annotations_df['label'] = annotations_df['label'].str.extract(r'(hate|noHate)$')
    
    txt_files = glob(os.path.join(data_folder, '*.txt'))
    
    data = []
    for txt_file in txt_files:
        file_name = os.path.basename(txt_file)
        file_id = file_name.split('.')[0]  # Remove .txt extension
        
        tweet_text = read_tweet_file(txt_file)
        if tweet_text is None:
            continue
            
        annotation = annotations_df[annotations_df['file_id'] == file_id]
        if not annotation.empty:
            label = annotation['label'].iloc[0]
            data.append({
                'file_id': file_id,
                'text': tweet_text,
                'label': label
            })
        else:
            logger.warning(f"No annotation found for file: {file_name}")
    
    df = pd.DataFrame(data)
    
    # Convert labels to binary
    df['label_binary'] = (df['label'] == 'hate').astype(int)
    
    return df

def prepare_datasets():
    """
    Prepare both training and test datasets.
    """
    os.makedirs('data', exist_ok=True)
    
    # Download and extract dataset
    download_and_extract_data(DATA_URL, EXTRACT_DIR)
    
    # Path to the extracted data (adjust this based on the folder structure after extraction)
    extracted_path = os.path.join(EXTRACT_DIR, 'hate-speech-dataset-master')
    
    # Process training data
    logger.info("Processing training data...")
    train_df = process_dataset(
        data_folder=os.path.join(extracted_path, 'sampled_train'),
        annotations_file=os.path.join(extracted_path,'annotations_metadata.csv')
    )
    
    # Process test data
    logger.info("Processing test data...")
    test_df = process_dataset(
        data_folder=os.path.join(extracted_path, 'sampled_test'),
        annotations_file=os.path.join(extracted_path, 'annotations_metadata.csv')
    )
    
    # Save processed datasets
    train_df.to_csv('data/processed_train.csv', index=False)
    test_df.to_csv('data/processed_test.csv', index=False)
    
    # Print dataset statistics
    logger.info(f"\nTraining samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    logger.info("\nLabel distribution in training set:")
    logger.info(train_df['label'].value_counts(normalize=True))
    
    return train_df, test_df

def validate_data(train_df, test_df):
    """
    Validate the processed datasets.
    """
    logger.info("\nValidating datasets...")
    
    train_empty = train_df['text'].isna().sum()
    test_empty = test_df['text'].isna().sum()
    
    if train_empty > 0:
        logger.warning(f"Found {train_empty} empty texts in training data")
    if test_empty > 0:
        logger.warning(f"Found {test_empty} empty texts in test data")
    
    logger.info("\nTraining set label distribution:")
    logger.info(train_df['label'].value_counts())
    logger.info("\nTest set label distribution:")
    logger.info(test_df['label'].value_counts())
    
    common_files = set(train_df['file_id']).intersection(set(test_df['file_id']))
    if common_files:
        logger.warning(f"Found {len(common_files)} files in both train and test sets!")

if __name__ == "__main__":
    # Prepare datasets
    train_df, test_df = prepare_datasets()
    
    # Validate the processed data
    validate_data(train_df, test_df)
    
    logger.info("\nData processing completed! Files saved as 'data/processed_train.csv' and 'data/processed_test.csv'")
