import pandas as pd

data = [
    {"question": "¿Qué carreras hay en la universidad?", "answer": "Ingeniería Química y Electrónica."},
    {"question": "¿Cuándo son las inscripciones?", "answer": "Las inscripciones son en marzo y septiembre."},
    {"question": "¿Cuáles son los requisitos de admisión?", "answer": "Se requiere presentar el examen de admisión y tener un promedio mínimo de 8.0 en preparatoria."},
    {"question": "¿Hay becas disponibles?", "answer": "Sí, la universidad ofrece becas por excelencia académica y por necesidades económicas."},
    {"question": "¿Cuánto cuesta la matrícula?", "answer": "La matrícula cuesta $5,000 por semestre."},
    {"question": "¿Dónde está ubicada la universidad?", "answer": "La universidad está ubicada en el centro de la ciudad."},
    {"question": "¿Qué programas de posgrado ofrecen?", "answer": "Ofrecemos maestrías en Administración y Ciencias de la Computación."},
    {"question": "¿Hay intercambios estudiantiles?", "answer": "Sí, tenemos convenios con universidades en Estados Unidos y Europa."},
    {"question": "¿Cuánto dura la carrera de Medicina?", "answer": "La carrera de Medicina dura seis años."},
    {"question": "¿Qué idiomas se enseñan?", "answer": "Enseñamos inglés, francés y alemán."}
]

df = pd.DataFrame(data)
df.to_csv('data/university_qa.csv', index=False)
