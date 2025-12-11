import os
import torch
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Force CPU
device = "cpu"

print("Loading Models... (This happens only once)")
# 1. Vision Model
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 2. Text Model (Flan-T5)
# Using 'base' for better creativity. Use 'small' if it lags.
text_model_name = "google/flan-t5-base" 
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForSeq2SeqLM.from_pretrained(text_model_name).to(device)
print("Models Ready!")

def get_factual_description(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=50)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def humanize_text(factual_caption, style):
    # This is "Few-Shot Prompting" - we show, don't just tell.
    
    if style == "social":
        # We give it examples of how to turn boring -> fun
        prompt = (
            "Task: Turn a boring image description into a fun, short, human-sounding Instagram caption with emojis.\n\n"
            "Example 1:\nInput: a dog sitting on the grass\nOutput: Just soaking up the sun! ☀️🐶 #GoodVibes\n\n"
            "Example 2:\nInput: a person drinking coffee\nOutput: Survival juice acquired. ☕️✨ #MondayMood\n\n"
            f"Input: {factual_caption}\nOutput:"
        )
    elif style == "poetic":
        prompt = (
            "Task: Rewrite this description as a short, moody, lowercase aesthetic sentence.\n"
            f"Input: {factual_caption}\nOutput:"
        )
    elif style == "roast":
        prompt = (
            "Task: Make a gentle, funny joke about this description.\n"
            f"Input: {factual_caption}\nOutput:"
        )
    else:
        return factual_caption

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        outputs = text_model.generate(
            input_ids, 
            max_new_tokens=60, 
            do_sample=True,   # CRITICAL: Enables creativity
            temperature=0.9,  # High = Creative, Low = Robotic
            top_p=0.95        # Nuance filter
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    final_caption = None
    filename = None
    style = "social"

    if request.method == 'POST':
        file = request.files.get('image')
        style = request.form.get('style')
        
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Step 1: Get the boring facts
            raw_desc = get_factual_description(filepath)
            
            # Step 2: Add the "Human" flavor
            final_caption = humanize_text(raw_desc, style)

    return render_template('index.html', caption=final_caption, filename=filename, style=style)

if __name__ == '__main__':
    app.run(debug=True)