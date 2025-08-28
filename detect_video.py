# Inside your script, after you get the 'results' from the model
for r in results:
    boxes = r.boxes  # Get the Boxes object
    for box in boxes:
        # Get the class ID (e.g., 0 for 'Ambulance')
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        # Your model's classes are {0: 'Ambulance', 1: 'items'}
        if class_id == 0 and confidence > 0.5: # If an ambulance is detected with > 50% confidence
            print(f"AMBULANCE DETECTED with {confidence*100:.2f}% confidence!")
            # --> ADD YOUR CUSTOM ACTION HERE <--
            # For example, send an alert, change a traffic light, etc.