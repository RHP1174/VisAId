import process_results as p
import threading
from describer import create_desc, tts
from process_results import process_results

def object_detection(video_data):
  # Create the description from the object data
  description = create_desc(video_data)

  # Output the description using TTS
  tts(description)

def main():
  # Simulate passing object detection results from your video processing
  # Replace this example with the actual object detection output

  video_data = process_results()

  # Start object detection and description generation in a separate thread
  detection_thread = threading.Thread(target=object_detection, args=(video_data,))

  # Start the thread
  detection_thread.start()

  # Join the thread to wait for it to complete
  detection_thread.join()

# Ensures that the main function is only executed when the script is run directly
if __name__ == "__main__":
  main()