# app.py
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
from models.GPT_2 import generate_text

# Función para actualizar la imagen, la descripción y el placeholder según la opción seleccionada
def update_image_description_placeholder(*args):
    option = option_var.get()
    if option == "BERT":
        image_path = "assets/BERT.png"
        description = "BERT es un modelo creado por Google que ha revolucionado el procesamiento del lenguaje natural..."
        placeholder = "¿Cual es la capital de Francia?"
    elif option == "DistilBERT":
        image_path = "assets/DistilBERT.png"
        description = "DistilBERT es una versión más ligera de BERT, diseñada para ser más eficiente en términos de velocidad y recursos..."
        placeholder = "Pregunta de ejemplo..."
    elif option == "OpenAI":
        image_path = "assets/OpenAI.png"
        description = "El modelo de OpenAI ha sido desarrollado para tareas avanzadas de generación de texto y comprensión del lenguaje..."
        placeholder = "Había una vez..."
    else:
        image_path = "assets/OpenAI.png"
        description = "Algo salió mal."
        placeholder = ""

    # Actualizar la imagen
    image = Image.open(image_path)
    image = image.resize((200, 200))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

    
    # Actualizar la descripción
    description_label.config(text=description)
    
    # Actualizar el placeholder del textbox
    textbox.delete(0, tk.END)
    textbox.insert(0, placeholder)
    textbox.bind("<FocusIn>", lambda event: on_focus_in(event, placeholder))
    textbox.bind("<FocusOut>", lambda event: on_focus_out(event, placeholder))

# Función para limpiar el placeholder al enfocar el textbox
def on_focus_in(event, placeholder):
    if textbox.get() == placeholder:
        textbox.delete(0, tk.END)

# Función para restaurar el placeholder si el textbox está vacío al perder el enfoque
def on_focus_out(event, placeholder):
    if textbox.get() == "":
        textbox.insert(0, placeholder)

# Función para abrir una nueva ventana con el contenido del textbox o ejecutar GPT-2 si la opción es "OpenAI"
def open_new_window():
    option = option_var.get()
    prompt = prompt_var.get()
    if option == "OpenAI":
        generated_text = generate_text(prompt)
        messagebox.showinfo("Respuesta", generated_text)
    elif option == "BERT":
        generated_text = generate_text(prompt)
        messagebox.showinfo("Respuesta", generated_text)
    elif option == "ASASA":
        generated_text = generate_text(prompt)
        messagebox.showinfo("Respuesta", generated_text)
    else:
        messagebox.showerror("Error", "El modelo seleccionado no existe o ha fallado")

# Función para centrar una ventana
def center_window(window):
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = ((screen_height - height) // 2) - 50
    window.geometry(f"{width}x{height}+{x}+{y}")

# Ventana principal
root = tk.Tk()
root.iconbitmap("assets/Atomic.ico")
root.title("Transformer Prototype")
root.geometry("500x600")
root.resizable(0,0)
center_window(root)

# Estilo global
default_font = ("Roboto", 12)
root.option_add("*Font", default_font)

# Frame para el combobox y su etiqueta
frame = tk.Frame(root)
frame.pack(pady=10)

# Etiqueta para el combobox
combobox_label = tk.Label(frame, text="Selecciona un Modelo:")
combobox_label.pack(side="left")

# Variable para almacenar la opción seleccionada
option_var = tk.StringVar()
option_var.trace("w", update_image_description_placeholder)

# Select (Combobox) con diferentes opciones
options = ["BERT", "DistilBERT", "OpenAI"]
combobox = ttk.Combobox(frame, state="readonly", textvariable=option_var, values=options)
combobox.current(0)
combobox.pack(side="left", padx=10)

# Label para mostrar la imagen
image_label = tk.Label(root)
image_label.pack(pady=10)

# Label para mostrar la descripción
description_label = tk.Label(root, wraplength=400, justify="left")
description_label.pack(pady=10)

# Textbox para ingresar el prompt
prompt_var = tk.StringVar()
textbox = tk.Entry(root, textvariable=prompt_var, width=int(root.winfo_width() * 0.8 / 10))  # Ajustar el ancho del textbox
textbox.pack(pady=10)

# Botón para abrir una nueva ventana con el contenido del textbox o ejecutar GPT-2 si la opción es "OpenAI"
button = tk.Button(root, text="Ejecutar", command=open_new_window)
button.pack(pady=10)

# Ejecutar la función para mostrar la imagen, la descripción y el placeholder iniciales
update_image_description_placeholder()

# Ejecutar la aplicación
root.mainloop()
