from transformers import T5ForConditionalGeneration, T5Tokenizer

# Cargar el modelo y el tokenizador fine-tuneados
model_name = "./fine_tuned_model"  # Asegúrate de que el directorio sea correcto
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def answer_question(question, context):
    max_length=100

    # Formatear la entrada según el formato esperado
    input_text = f"question: {question} context: {context}"
    
    # Tokenizar la entrada
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generar respuesta a partir del input
    outputs = model.generate(
        inputs['input_ids'], 
        max_length=max_length, 
        num_beams=4, 
        early_stopping=True
    )

    # Decodificar la respuesta generada
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Ejemplo de uso (No necesario si se runnea el app.py)
if __name__ == "__main__":
    while True:
        context = input("Introduce un contexto: ")
        question = input("Introduce una pregunta: ")
        if question.lower() in ['salir', 'exit', 'quit']:
            break
        answer = answer_question(question, context)
        print("Respuesta generada:", answer)
