from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-large")

# Define your prompt (the code or comment to begin with)
prompt = "Is 'open source data science' a thing? Are there any interesting public data sets lying around and are there examples of people finding 'gems' in such sets?"

# Tokenize the input prompt
inputs = tokenizer(prompt, return_tensors="pt")

# Generate code with a large number of tokens
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)

# Decode the generated tokens into text (the code)
generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("------------------------------ Results ------------------------------")
print(generated_code)
