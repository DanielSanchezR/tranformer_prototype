import json

input_file = 'data/sabana_qa_dataset.json'
output_file = 'data/formatted_sabana_qa_dataset.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

formatted_data = []
for item in data:
    formatted_data.append({
        'input': item['question'],
        'target': item['answer']
    })

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)
