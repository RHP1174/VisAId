from openvino_genai import LLMPipeline
import pyttsx3

def create_desc(video_data):
  # Construct a description prompt based on the video data (list of tuples)
  if not video_data:
    return "No objects detected in the scene."

  # Create a string that describes the surroundings
  object_descriptions = []
  for obj_name, obj_count in video_data:
    # Singular or plural depending on the count
    if obj_count == 1:
      object_descriptions.append(f"1 {obj_name}")
    else:
      object_descriptions.append(f"{obj_count} {obj_name}s")

  # Join the object descriptions to form a full scene description
  scene_description = "In the surroundings, there are " + ", ".join(object_descriptions) + "."

  # Use the scene description in the LLM prompt
  prompt = f"Based on the following scene description: '{scene_description}', write a detailed description of the surroundings."

  pipe = LLMPipeline()
  res = pipe.generate(prompt)
  return res

def tts(text):
  # Initialize pyttsx3 engine for Text-to-Speech
  engine = pyttsx3.init()

  # Optionally, set properties like speed and volume
  engine.setProperty('rate', 150)  # Speed of speech
  engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)

  # Speak the text
  engine.say(text)
  engine.runAndWait()