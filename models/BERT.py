# Models/BERT_QA.py
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Cargar el modelo y el tokenizador
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def answer_question_BERT(question, context):
    # Tokenizar entrada
    inputs = tokenizer(question, context, return_tensors='pt')
    input_ids = inputs['input_ids']
    token_type_ids = inputs['token_type_ids']

    # Obtener puntuaciones de inicio y final
    outputs = model(input_ids, token_type_ids=token_type_ids)
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Obtener posiciones de inicio y final con las puntuaciones más altas
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    # Convertir tokens en la respuesta
    answer_tokens = input_ids[0][start_index:end_index]
    answer = tokenizer.decode(answer_tokens)
    
    return answer

# Ejemplo de uso (No necesario si se runnea el app.py)
if __name__ == "__main__":
    context = "France is a country in Europe. The capital of France is Paris."
    while True:
        question = input("Introduce una pregunta: ")
        if question.lower() in ['salir', 'exit', 'quit']:
            break
        answer = answer_question_BERT(question, context)
        print("Respuesta:", answer)
