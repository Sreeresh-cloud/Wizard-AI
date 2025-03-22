import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, get_scheduler
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load BLIP model from local checkpoint
model_path = "./models/blip_finetuned"
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
model.to(device)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

print("Model and processor loaded successfully!")

# Load dataset
dataset_path = "./dataset/dataset.json"

class CaptionDataset(Dataset):
    def __init__(self, json_path, processor):
        with open(json_path, "r") as f:
            self.data = json.load(f)  # Ensure it's a list of dictionaries

        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        text = item["caption"]

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, text=text, return_tensors="pt", padding="max_length", max_length=50)

        return {k: v.squeeze(0) for k, v in inputs.items()}

# Create dataloader
dataset = CaptionDataset(dataset_path, processor)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Dataset loaded successfully!")

# Apply LoRA (Low-Rank Adaptation)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,  # Scaling factor
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("LoRA applied successfully!")

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3  # Assuming 3 epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)

print("Optimizer and scheduler set up!")

# Training loop
num_epochs = 6

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()
        outputs = model(**batch, labels=batch["input_ids"])  # Pass labels
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} completed - Avg Loss: {avg_loss:.4f}")

# Save fine-tuned model
save_path = "./models/blip_finetuned"
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print("Fine-tuned model saved successfully at:", save_path)
