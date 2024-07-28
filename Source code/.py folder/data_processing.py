import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, model, device):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
        hidden_states = outputs.hidden_states
        second_to_last_layer = hidden_states[-2]
        sentence_embedding = second_to_last_layer.squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long, device=self.device),
            'embedding': sentence_embedding
        }

def load_data(file_path, model_name, device):
    df = pd.read_csv(file_path)
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(df['text'].values, df['emotion'].values, test_size=0.2, random_state=42)
    val_texts, test_texts, val_labels, test_labels = train_test_split(temp_texts, temp_labels, test_size=0.5, random_state=42)

    label_encoder = LabelEncoder()
    label_encoder.fit(df['emotion'])

    tokenizer = AutoTokenizer.from_pretrained(model_name, keep_accents=True)

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_encoder, tokenizer
