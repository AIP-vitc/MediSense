import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# 1. Load the heavy BioBERT model
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
model.eval()

# 2. Save the Tokenizer in a standalone JSON format
# This removes the need for the transformers library in the user file
tokenizer.save_pretrained("./offline_model")

# 3. Export the Model to ONNX format
dummy_input = torch.ones(1, 128, dtype=torch.long)
torch.onnx.export(
    model, 
    (dummy_input, dummy_input), 
    "./offline_model/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}, 
                  'attention_mask': {0: 'batch', 1: 'sequence'}},
    opset_version=14
)

print("✅ Success! Move the 'offline_model' folder to your production machine.")