import cv2
import time
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Easily accessible configuration variables
CONFIDENCE_THRESHOLD = 0.85  # Confidence threshold for letter detection (increased to reduce false positives)
LETTER_COOLDOWN = 0.8  # Seconds to wait before accepting the same letter again (increased to reduce duplicates)
STABLE_THRESHOLD = 8  # Number of consecutive frames to consider a letter stable (increased for more stability)

class SignLanguageTranslator:
    def __init__(self, confidence_threshold=CONFIDENCE_THRESHOLD):
        print("Initializing Sign Language Translator...")

        # Set the confidence threshold (easily modifiable parameter)
        self.confidence_threshold = confidence_threshold
        print(f"Confidence threshold set to: {self.confidence_threshold}")

        # Load the model
        print("Loading model...")
        try:
            # Try to load the model from .h5 file
            self.model = keras.models.load_model('model.h5')
            print("Model loaded from model.h5")
        except Exception as e:
            print(f"Error loading model from .h5: {e}")
            try:
                # Try to load the model from JSON and weights
                with open('model.json', 'r') as json_file:
                    model_json = json_file.read()
                self.model = keras.models.model_from_json(model_json)
                print("Model loaded from model.json")
            except Exception as e2:
                print(f"Error loading model from JSON: {e2}")
                raise ValueError("Failed to load model from any available source")

        # Create a mapping for the output labels (A-Z excluding J and Z)
        self.robust_id2label = {}
        for i in range(24):  # 24 letters (A-Y excluding J and Z)
            if i < 9:  # A-I
                self.robust_id2label[i] = chr(65 + i)
            else:  # K-Y (skipping J)
                self.robust_id2label[i] = chr(65 + i + 1)

        print("\nLabel mapping:")
        for idx, label in sorted(self.robust_id2label.items()):
            print(f"  {idx}: {label}")

        # Initialize variables for word formation
        self.current_word = []
        self.words_history = []
        self.last_letter = None
        self.last_letter_time = 0
        self.letter_cooldown = LETTER_COOLDOWN
        self.stable_letter = None
        self.stable_letter_count = 0
        self.stable_threshold = STABLE_THRESHOLD

        # Initialize camera
        print("Setting up camera...")
        self.cap = cv2.VideoCapture(0)

        # Try multiple camera indices if the first one doesn't work
        if not self.cap.isOpened():
            print("Camera index 0 failed, trying index 1...")
            self.cap = cv2.VideoCapture(1)

        if not self.cap.isOpened():
            print("Camera index 1 failed, trying index 2...")
            self.cap = cv2.VideoCapture(2)

        if not self.cap.isOpened():
            raise ValueError("Could not open webcam. Please check your camera connection.")

        # Set up window
        print("Setting up display window...")
        cv2.namedWindow("Sign Language Translator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Sign Language Translator", 800, 600)

        print("Initialization complete!")

    def preprocess_image(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract the region of interest (center of the frame)
        h, w = rgb_frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        roi_size = 200  # Size of the region of interest

        # Calculate ROI coordinates
        x1 = max(0, center_x - roi_size // 2)
        y1 = max(0, center_y - roi_size // 2)
        x2 = min(w, center_x + roi_size // 2)
        y2 = min(h, center_y + roi_size // 2)

        # Extract ROI
        roi = rgb_frame[y1:y2, x1:x2]

        # Convert to grayscale (since sign language models often work with grayscale)
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        # Resize to match the expected input dimensions (30, 63)
        resized = cv2.resize(gray, (63, 30))

        # Normalize pixel values to [0, 1]
        normalized = resized / 255.0

        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)

        return input_data

    def predict_letter(self, frame):
        input_data = self.preprocess_image(frame)

        # Make prediction
        predictions = self.model.predict(input_data, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])

        # Calculate confidence
        confidence = float(predictions[0][predicted_class_idx])

        # Get the predicted letter using our robust mapping
        if predicted_class_idx in self.robust_id2label:
            predicted_letter = self.robust_id2label[predicted_class_idx]
        else:
            # If the index is out of range, use a default value
            predicted_letter = '?'
            # Lower the confidence for unknown predictions
            confidence = confidence * 0.5

        return predicted_letter, confidence

    def add_letter_to_word(self, letter, confidence):
        current_time = time.time()

        # Only add letter if confidence is high enough
        if confidence < self.confidence_threshold:
            return False

        # Check if it's the same letter as before and if enough time has passed
        if letter == self.last_letter and current_time - self.last_letter_time < self.letter_cooldown:
            return False

        # Update last letter info
        self.last_letter = letter
        self.last_letter_time = current_time

        # Add letter to current word
        self.current_word.append(letter)
        return True

    def complete_word(self):
        if not self.current_word:
            return

        # Join letters to form a word
        word = ''.join(self.current_word)

        # Add to history
        self.words_history.append(word)

        # Clear current word
        self.current_word = []

    def clear_current_word(self):
        self.current_word = []

    def clear_history(self):
        self.words_history = []

    def draw_ui(self, frame, letter, confidence, letter_added=False):
        # Create a copy of the frame to draw on
        display = frame.copy()

        # Draw rectangle for letter display - green if confidence is high, red if low
        rect_color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 0, 255)
        cv2.rectangle(display, (10, 10), (100, 100), rect_color, 2)

        # Display the current letter - make it bigger and bolder
        cv2.putText(display, letter, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        # Display confidence with color based on threshold
        conf_color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 0, 255)
        cv2.putText(display, f"Conf: {confidence:.2f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1)

        # Display current threshold value
        cv2.putText(display, f"Threshold: {self.confidence_threshold:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display stability counter
        if self.stable_letter == letter and confidence >= self.confidence_threshold:
            stability_text = f"Stability: {self.stable_letter_count}/{self.stable_threshold}"
            cv2.putText(display, stability_text, (120, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display current word being formed with highlight for the last letter
        current_word_str = ''.join(self.current_word)
        cv2.putText(display, f"Current word: {current_word_str}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # If a letter was just added, show a visual indicator
        if letter_added:
            cv2.putText(display, "Letter Added!", (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display word history (last 3 words)
        history_display = ' '.join(self.words_history[-3:])
        cv2.putText(display, f"History: {history_display}", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw a guide box in the center to help position the hand
        cv2.rectangle(display,
                     (display.shape[1]//2 - 100, display.shape[0]//2 - 100),
                     (display.shape[1]//2 + 100, display.shape[0]//2 + 100),
                     (255, 255, 255), 1)

        # Display instructions
        cv2.putText(display, "Press SPACE to complete word", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "Press BACKSPACE to delete last letter", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "Press C to clear current word", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "Press H to clear history", (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "Press + to increase threshold", (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "Press - to decrease threshold", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "Press Q to quit", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return display

    def run(self):
        print("Starting sign language translation...")
        print("Press SPACE to complete a word")
        print("Press BACKSPACE to delete the last letter")
        print("Press C to clear the current word")
        print("Press H to clear the word history")
        print("Press + to increase confidence threshold")
        print("Press - to decrease confidence threshold")
        print("Press Q to quit")

        try:
            while True:
                # Capture frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                # Flip the frame horizontally for a more natural interaction
                frame = cv2.flip(frame, 1)

                # Predict letter
                letter, confidence = self.predict_letter(frame)

                # Check for stable letter detection
                letter_added = False
                if confidence >= self.confidence_threshold:
                    if letter == self.stable_letter:
                        self.stable_letter_count += 1
                        if self.stable_letter_count >= self.stable_threshold:
                            # Letter is stable for enough frames, add it to the word
                            letter_added = self.add_letter_to_word(letter, confidence)
                            # Reset stability counter after adding
                            if letter_added:
                                self.stable_letter_count = 0
                                print(f"Added letter: {letter} (confidence: {confidence:.2f})")
                    else:
                        # New letter detected, reset stability counter
                        self.stable_letter = letter
                        self.stable_letter_count = 1
                else:
                    # Low confidence, reset stability counter
                    self.stable_letter_count = 0

                # Draw UI with feedback about letter detection
                display = self.draw_ui(frame, letter, confidence, letter_added)

                # Show the frame
                cv2.imshow("Sign Language Translator", display)

                # Process key presses
                key = cv2.waitKey(1) & 0xFF

                # Key controls
                if key == ord(' '):  # Space to complete word
                    self.complete_word()
                    print(f"Completed word: {''.join(self.current_word)}")
                elif key == 8:  # Backspace to delete last letter
                    if self.current_word:
                        removed = self.current_word.pop()
                        print(f"Removed letter: {removed}")
                elif key == ord('c'):  # C to clear current word
                    self.clear_current_word()
                    print("Cleared current word")
                elif key == ord('h'):  # H to clear history
                    self.clear_history()
                    print("Cleared history")
                elif key == ord('+') or key == ord('='):  # + to increase threshold
                    self.confidence_threshold = min(1.0, self.confidence_threshold + 0.01)
                    print(f"Increased confidence threshold to: {self.confidence_threshold:.2f}")
                elif key == ord('-') or key == ord('_'):  # - to decrease threshold
                    self.confidence_threshold = max(0.01, self.confidence_threshold - 0.01)
                    print(f"Decreased confidence threshold to: {self.confidence_threshold:.2f}")
                elif key == ord('q'):  # Q to quit
                    break

        finally:
            # Release resources
            self.cap.release()
            cv2.destroyAllWindows()

            # Print final translation
            print("\nFinal Translation:")
            if self.words_history:
                print(" ".join(self.words_history))
            else:
                print("No words translated.")


if __name__ == "__main__":
    try:
        # You can easily modify the confidence threshold here
        translator = SignLanguageTranslator(confidence_threshold=CONFIDENCE_THRESHOLD)
        translator.run()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print("\nDetailed error information:")
        traceback.print_exc()
