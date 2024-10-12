import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from IPython.display import display, Image as IPImage
from huggingface_hub import login
import os
import time

# Hugging Face authentication
login(token="")

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
cache_dir = "./model_cache"  # Specify a cache directory

# Check if model files already exist
if os.path.exists(cache_dir) and any(file.endswith('.bin') for file in os.listdir(cache_dir)):
    print("Using cached model files")
else:
    print("Downloading model files")

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    device_map="auto",
    cache_dir=cache_dir,
)
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)

# Function to extract text from an image
def extract_text_from_image(image, prompt):
    # Prepare the input prompt
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process the input
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(device)

    # Generate the output
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        output = model.generate(**inputs, max_new_tokens=200)
    
    # Decode and return the result
    return processor.decode(output[0], skip_special_tokens=True)

# Function to process image from a given path
def process_image_from_path(image_path, prompt):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return
    
    # Open and display the image
    image = Image.open(image_path)
    display(IPImage(filename=image_path))
    
    print("Processing image...")
    
    # Start the timer
    start_time = time.time()
    
    # Process the image
    extracted_text = extract_text_from_image(image, prompt)
    
    # End the timer
    end_time = time.time()
    processing_time = end_time - start_time
    
    # Display the result
    print(f"Prompt: {prompt}")
    print(f"Extracted Text:\n{extracted_text}")
    print(f"\nProcessing Time: {processing_time:.2f} seconds")

# Interactive loop for inferencing
def interactive_inference():
    while True:
        image_path = input("\nEnter the path to an image (or 'quit' to exit): ")
        if image_path.lower() == 'quit':
            break
        prompt = input("Enter your prompt (press Enter for default): ")
        if not prompt:
            prompt = "Please read and transcribe all the text you can see in this image."
        process_image_from_path(image_path, prompt)

# Run the interactive inference loop
if __name__ == "__main__":
    print("Model loaded. Ready for inferencing.")
    interactive_inference()