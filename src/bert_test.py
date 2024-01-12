from transformers import BertModel, BertTokenizer
import torch

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
text = "Il tuo testo da elaborare."
encoded_text = tokenizer.encode(text, add_special_tokens=True)
input_ids = torch.tensor([encoded_text])
with torch.no_grad():
    outputs = model(input_ids)
    print("Successfully")
    print(outputs.last_hidden_state)
