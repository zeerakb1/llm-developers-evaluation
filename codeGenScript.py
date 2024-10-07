from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")  # You can use a different version if needed
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")

prompt = "Is 'open source data science' a thing? Are there any interesting public data sets lying around and are there examples of people finding 'gems' in such sets?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)

generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("------------------------------ Results ------------------------------")
print(generated_code)