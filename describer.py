from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          pipeline)
import pyttsx3
import json

config_data = json.load(open("config.json"))
HF_TOKEN = config_data["HF_TOKEN"]

model_name = "meta-llama/Llama-3.2-1B"

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          token=HF_TOKEN)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
  model_name,
  device_map="auto",
  token=HF_TOKEN
)

text_generator = pipeline(
  "text-generation",
  model=model,
  tokenizer=tokenizer,
  return_full_text=False,
  max_new_tokens=60,
  temperature = .25
)


def create_desc(video_data):
  # Construct a description prompt based on the video data (list of tuples)
  if not video_data:
    return "No objects detected in the scene."

  # Create a string that describes the surroundings
  object_descriptions = []
  for obj_name, obj_count in video_data.items():
    # Singular or plural depending on the count
    if obj_count == 1:
      object_descriptions.append(f"1 {obj_name}")
    else:
      object_descriptions.append(f"{obj_count} {obj_name}s")

  # Join the object descriptions to form a full scene description
  scene_description = "In the surroundings, there are " + ", ".join(object_descriptions) + "."

  # Use the scene description in the LLM prompt
  prompt = f"Based on the following description of a picture: '{scene_description}', describe the environment you have gotten and what is in the environment."

  sequences = text_generator(prompt)
  gen_text = sequences[0]["generated_text"]
  print(gen_text)
  return gen_text

def tts(text):
  # Initialize pyttsx3 engine for Text-to-Speech
  engine = pyttsx3.init()

  # Optionally, set properties like speed and volume
  engine.setProperty('rate', 150)  # Speed of speech
  engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

  # Speak the text
  engine.say(text)
  engine.runAndWait()
