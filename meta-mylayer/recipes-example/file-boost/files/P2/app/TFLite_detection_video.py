import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter

def tflite_detect_video(model_path, video_path, lbl_path, min_conf=0.7, output_path='output_video.mp4'):
    # Load the label map
    with open(lbl_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the TensorFlow Lite model into memory
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Open the video file
    #cap = cv2.VideoCapture(video_path)
    # Use Webcam
    cap = cv2.VideoCapture(0)
    
    # Get the video codec and create a VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to 640x480
        frame_resized = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Prepare the image for the model's input size
        input_data = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(input_data, axis=0)

        # Normalize pixel values if using a float model
        float_input = input_details[0]['dtype'] == np.float32
        input_mean = 127.5
        input_std = 127.5
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get the results
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]  # Bounding box coordinates
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Class index
        scores = interpreter.get_tensor(output_details[0]['index'])[0]  # Confidence scores

        # Loop over each detection
        for i in range(len(scores)):
            if min_conf < scores[i] <= 1.0:
                ymin = int(max(1, (boxes[i][0] * 480)))  # Adjust for resized frame
                xmin = int(max(1, (boxes[i][1] * 640)))
                ymax = int(min(480, (boxes[i][2] * 480)))
                xmax = int(min(640, (boxes[i][3] * 640)))

                # Draw the bounding box
                cv2.rectangle(frame_resized, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Label the object
                object_name = labels[int(classes[i])]
                label = f'{object_name}: {int(scores[i] * 100)}%'
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame_resized, (xmin, label_ymin - label_size[1] - 10), 
                              (xmin + label_size[0], label_ymin + baseline - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(frame_resized, label, (xmin, label_ymin - 7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # Write the frame with detections to the output video
        out.write(frame_resized)

        # Display the frame with detections
        cv2.imshow('Detections', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
model_path = 'rock_detect.tflite'  # Path to your TFLite model
video_path = 'rock_mars.mp4'       # Path to your input video
lbl_path = 'rock_label_map.pbtxt'  # Path to your labels file
tflite_detect_video(model_path, video_path, lbl_path)

