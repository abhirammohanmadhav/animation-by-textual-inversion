import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import cv2
import numpy as np
import subprocess

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

def generate_text(prompt, length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(
        input_ids=input_ids,
        max_length=length + len(input_ids[0]),
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        num_return_sequences=1,
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text[len(prompt) :]

start_prompt = "A photo of <ugly-sonic>"
end_prompt = "A photo of <cheburashka>"

intermediate_prompts = []
for i in range(1, 20):
    alpha = i / 20.0
    prompt = f"A photo of <{generate_text(start_prompt[12:-1])}>"
    prompt += f" interpolates into A photo of <{generate_text(end_prompt[12:-1])}>"
    intermediate_prompts.append(prompt)

print(intermediate_prompts)



# Set up OpenCV video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
width, height = 640, 480
fps = 24
video_writer = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# Generate images from prompts and add them to the video
for prompt in intermediate_prompts:
    # Generate image from prompt
    text_img = np.zeros((height, width, 3), np.uint8)
    text_img[:] = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(prompt, font, font_scale, font_thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height - text_size[1]) // 2
    cv2.putText(text_img, prompt, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)

    # Add image to video
    video_writer.write(text_img)

# Release video writer and finalize video file
video_writer.release()

# Compress video using ffmpeg
subprocess.run(["ffmpeg", "-i", "output.mp4", "-vf", "scale=320:-2", "-c:a", "copy", "compressed_output.mp4"])


