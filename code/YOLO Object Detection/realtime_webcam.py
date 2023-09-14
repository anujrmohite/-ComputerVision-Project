import numpy as np
import cv2

# Initialize the webcam video stream (0 represents the default camera)
webcam_video_stream = cv2.VideoCapture(0)

while True:
    # Read the current frame from the webcam
    ret, current_frame = webcam_video_stream.read()
    img_to_detect = current_frame
    img_height = img_to_detect.shape[0]
    img_width = img_to_detect.shape[1]

    # Prepare the image for the YOLO model by creating a blob
    img_blob = cv2.dnn.blobFromImage(img_to_detect, 0.003922, (320, 320), swapRB=True, crop=False)

    # List of class labels used by YOLO model
    class_labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                    "trafficlight", "firehydrant", "stopsign", "parkingmeter", "bench", "bird", "cat",
                    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sportsball",
                    "kite", "baseballbat", "baseballglove", "skateboard", "surfboard", "tennisracket",
                    "bottle", "wineglass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hotdog", "pizza", "donut", "cake", "chair",
                    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
                    "remote", "keyboard", "cellphone", "microwave", "oven", "toaster", "sink", "refrigerator",
                    "book", "clock", "vase", "scissors", "teddybear", "hairdrier", "toothbrush"]

    # Define colors for bounding boxes
    class_colors = ["0,255,0", "0,0,255", "255,0,0", "255,255,0", "0,255,255"]
    class_colors = [np.array(every_color.split(",")).astype("int") for every_color in class_colors]
    class_colors = np.array(class_colors)
    class_colors = np.tile(class_colors, (16, 1))

    # Load the YOLO model from configuration and weights files
    yolo_model = cv2.dnn.readNetFromDarknet('/home/babayaga/Documents/OCR/Optical Character Recognition, Image Recognition Object Detection and Object Recognition/dataset/yolov3.cfg',
                                            '/home/babayaga/Documents/OCR/Optical Character Recognition, Image Recognition Object Detection and Object Recognition/dataset/yolov3.weights')

    # Get the output layers of the YOLO model
    yolo_layers = yolo_model.getLayerNames()
    yolo_output_layer = [yolo_layers[yolo_layer - 1] for yolo_layer in yolo_model.getUnconnectedOutLayers()]

    # Set the input for the YOLO model and get detection layers
    yolo_model.setInput(img_blob)
    obj_detection_layers = yolo_model.forward(yolo_output_layer)

    # Iterate through the detection layers
    for object_detection_layer in obj_detection_layers:
        for object_detection in object_detection_layer:

            # Extract confidence scores and class predictions
            all_scores = object_detection[5:]
            predicted_class_id = np.argmax(all_scores)
            prediction_confidence = all_scores[predicted_class_id]

            # Filter detections with confidence greater than 50%
            if prediction_confidence > 0.50:
                predicted_class_label = class_labels[predicted_class_id]

                # Calculate bounding box coordinates
                bounding_box = object_detection[0:4] * np.array([img_width, img_height, img_width, img_height])
                (box_center_x_pt, box_center_y_pt, box_width, box_height) = bounding_box.astype("int")
                start_x_pt = int(box_center_x_pt - (box_width / 2))
                start_y_pt = int(box_center_y_pt - (box_height / 2))
                end_x_pt = start_x_pt + box_width
                end_y_pt = start_y_pt + box_height

                box_color = class_colors[predicted_class_id]
                box_color = [int(c) for c in box_color]

                predicted_class_label = "{}: {:.2f}%".format(predicted_class_label, prediction_confidence * 100)
                print("predicted object {}".format(predicted_class_label))

                # Draw bounding box and label on the image
                cv2.rectangle(img_to_detect, (start_x_pt, start_y_pt), (end_x_pt, end_y_pt), box_color, 1)
                cv2.putText(img_to_detect, predicted_class_label, (start_x_pt, start_y_pt - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

    # Display the processed frame
    cv2.imshow("Detection Output", img_to_detect)

    # Terminate the loop if 'q' or 'quit' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close all OpenCV windows
webcam_video_stream.release()
cv2.destroyAllWindows()
