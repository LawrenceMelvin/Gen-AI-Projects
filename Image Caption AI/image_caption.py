import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Load the pretrained processor and model
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image, DONT FORGET TO WRITE YOUR IMAGE NAME
img_path = "C:\\Lawrence\\Gen AI Projects\\Image Caption AI\\nature.jpg"
# convert it into an RGB format
image = Image.open(img_path).convert('RGB')


# You do not need a question for image captioning
text = "the image of"
inputs = processor(images=image, text=text, return_tensors="pt") #pt-pytorch-tensor

# Generate a caption for the image
outputs = model.generate(**inputs, max_length=50)

"""
Finally, the generated output is a sequence of tokens. 
To transform these tokens into human-readable text, 
you use the decode method provided by the processor. 
The skip_special_tokens argument is set to True to ignore special tokens in the output text.
"""
# Decode the generated tokens to text
caption = processor.decode(outputs[0], skip_special_tokens=True)
# Print the caption
print(caption)