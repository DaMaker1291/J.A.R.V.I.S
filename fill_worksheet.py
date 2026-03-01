#!/usr/bin/env python3
"""
Script to copy handwriting from filled page and generate styled text for new page
"""

import sys
import os
sys.path.append('/Users/shaurjeshbasu/CascadeProjects/windsurf-project-6')

import yaml
from jason.modules.general_assistant import GeneralAssistant

def main():
    # Load config from config.yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set gemini api key
    config['gemini_api_key'] = config.get('api_keys', {}).get('gemini', '')
    
    if not config['gemini_api_key']:
        print("Error: Gemini API key not found in config.yaml")
        return
    
    assistant = GeneralAssistant(config)
    
    # Paths to images
    filled_image_path = '/Users/shaurjeshbasu/CascadeProjects/windsurf-project-6/Screenshot 2026-02-19 at 11.42.56.png'
    new_page_path = '/Users/shaurjeshbasu/CascadeProjects/windsurf-project-6/Screenshot 2026-02-19 at 11.43.47.png'
    
    # Load images
    import PIL.Image
    filled_image = PIL.Image.open(filled_image_path)
    new_page_image = PIL.Image.open(new_page_path)
    
    # Extract text from filled page
    text_response = assistant.model.generate_content(["Extract all handwritten text from this image. Return only the text, no explanations.", filled_image])
    extracted_text = text_response.text.strip()
    
    print(f"Extracted text: {extracted_text}")
    
    # Get bounding boxes for handwritten regions
    box_response = assistant.model.generate_content(["Identify all regions with handwritten text in this image. Return a JSON object with 'boxes' key containing an array of [x1,y1,x2,y2] where coordinates are pixels from top-left (0,0). Return only the JSON object.", filled_image])
    box_text = box_response.text.strip()
    print(f"Bounding boxes: {box_text}")
    
    try:
        import json
        data = json.loads(box_text)
        boxes = data['boxes']
    except:
        print("Failed to parse bounding boxes")
        boxes = []
    
    # Crop and save handwritten regions
    cropped_images = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        try:
            cropped = filled_image.crop((x1, y1, x2, y2))
            crop_path = f'/Users/shaurjeshbasu/CascadeProjects/windsurf-project-6/cropped_handwriting_{i}.png'
            cropped.save(crop_path)
            cropped_images.append(crop_path)
            print(f"Cropped handwriting region saved to: {crop_path}")
        except Exception as e:
            print(f"Failed to crop region {i}: {e}")
    
    # Open all cropped images for copying
    for path in cropped_images:
        subprocess.run(["open", path])
    
    print("Cropped handwriting images opened. Copy and paste each into the new worksheet page.")

if __name__ == "__main__":
    main()
