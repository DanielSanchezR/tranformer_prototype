# models/test_bert.py
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Cargar el modelo y el tokenizador ajustados
model_name = "models/bert-finetuned-university-qa"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

def answer_question(question, context):
    # Tokenizar las entradas
    inputs = tokenizer.encode_plus(question, context, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]

    # Obtener las salidas del modelo
    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=token_type_ids)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

    # Obtener las posiciones de inicio y fin de la respuesta
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores) + 1

    # Decodificar la respuesta
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_index:end_index]))
    return answer

# Ejemplo de uso
if __name__ == "__main__":
    context = "La universidad está ubicada en el centro de la ciudad. Las inscripciones son en marzo y septiembre. Ofrecemos maestrías en Administración y Ciencias de la Computación. Sí, la universidad ofrece becas por excelencia académica y por necesidades económicas."
    
    while True:
        question = input("Introduce una pregunta: ")
        if question.lower() in ['salir', 'exit', 'quit']:
            break
        answer = answer_question(question, context)
        print("Respuesta:", answer)
