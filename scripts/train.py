import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

# Cargar el dataset
df = pd.read_csv('data/university_qa.csv')

# Renombrar las columnas a "text" y "labels"
df = df.rename(columns={"question": "text", "answer": "labels"})

# Convertir a dataset de Hugging Face
dataset = Dataset.from_pandas(df)

# Inicializar el tokenizador
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Función de preprocesamiento
def preprocess_function(examples):
    inputs = [q for q in examples["text"]]
    targets = [a for a in examples["labels"]]
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=128)
    labels = tokenizer(targets, padding="max_length", truncation=True, max_length=128)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Aplicar preprocesamiento
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Cargar el modelo
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(df["labels"].unique()))

# Configurar los argumentos de entrenamiento
training_args = TrainingArguments(
    output_dir="models/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Inicializar el entrenador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Entrenar el modelo
trainer.train()
