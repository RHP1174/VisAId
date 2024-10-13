import process_results as p
import threading
from describer import create_desc, tts

def object_detection(video_data):
  description = create_desc(video_data)
  tts(description)

def main():
  video_data = p.process_results()
  print("Object Counts:", video_data)
  detection_thread = threading.Thread(target=object_detection, args=(video_data,))
  detection_thread.start()
  detection_thread.join()

if __name__ == "__main__":
  main()