from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import os

model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

folder = input("Enter the folder path (press Enter for default 'image'): ") or "images"

if not os.path.exists(folder):
    print("⚠️ Folder not found. Please check the path and try again")
    exit()

files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
if not files:
    print("⚠️ No images found in this folder.")
    exit()

captions = []
for file in files:
    path = os.path.join(folder, file) 
    try:
        image = Image.open(path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=16, num_beams = 4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        captions.append((file, caption))
        print(f"{file}: {caption}")
    except Exception as e:
        print(f"❌ Error processing {file}: {e}")

with open("captions_summary.txt", "w") as f:
    for name, cap in captions:
        f.write(f"{name}: {cap}\n")  

print("\n✅ All captions saved to 'captions_summary.txt'")               

