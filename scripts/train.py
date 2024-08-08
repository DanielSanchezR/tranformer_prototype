import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments

# Cargar dataset
with open("data/sabana_qa_dataset.json") as f:
    dataset = json.load(f)['data']

# Preprocesamiento del dataset
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qa = self.data[idx]
        input_text = f"question: {qa['question']} context: {qa['context']}"
        target_text = qa['answers'][0]

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        targets = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

# Tokenización
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# Preparar Dataset y DataLoader
train_dataset = QADataset(dataset, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Cargar modelo preentrenado
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# Configuración del entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Entrenamiento
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# Guardar modelo fine-tuneado
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
