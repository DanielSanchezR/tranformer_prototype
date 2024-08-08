# Models/GPT-2.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar el modelo y el tokenizador
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_text(prompt, max_length=200):
    # Tokenizar el prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    # Generar texto a partir del prompt
    outputs = model.generate(input_ids, do_sample=True, temperature=0.9, attention_mask=None, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Ejemplo de uso (No necesario si se runnea el app.py)
if __name__ == "__main__":
    while True:
        prompt = input("Introduce un prompt: ")
        if prompt.lower() in ['salir', 'exit', 'quit']:
            break
        generated_text = generate_text(prompt)
        print("Texto generado:", generated_text)
