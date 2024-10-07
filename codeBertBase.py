from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForMaskedLM.from_pretrained("microsoft/codebert-base")

# Define your prompt with a masked token
prompt = "Is 'open source data science' a thing? Are there any interesting public data sets lying around and are there examples of people finding 'gems' in such sets? [MASK]"

# Tokenize the input, including the mask token
inputs = tokenizer(prompt, return_tensors="pt")

# Predict the masked token
outputs = model(**inputs)
logits = outputs.logits

# Get the predictions for the [MASK] token
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
predicted_token = tokenizer.decode(predicted_token_id)

# Print the predicted token
print(f"Predicted token: {predicted_token}")
