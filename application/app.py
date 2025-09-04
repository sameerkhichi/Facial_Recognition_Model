import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
import tensorflow as tf
import numpy as np
from

# Load Siamese model
model = tf.keras.models.load_model(
    "siamesemodelv2.h5",
    custom_objects={"L1Dist": L1Dist, "BinaryCrossentropy": tf.losses.BinaryCrossentropy}
)

verification_dir = "live_test_data/verification_images"
os.makedirs(verification_dir, exist_ok=True)

def capture_image(path):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(path, frame)
    cap.release()

def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (100, 100))
    img = img / 255.0
    return img

def add_user():
    name = simpledialog.askstring("Input", "Enter user name:")
    if not name:
        return

    user_dir = os.path.join(verification_dir, name)
    os.makedirs(user_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)

    instructions = [
        "Look straight",
        "Tilt head left",
        "Tilt head right",
        "Look up",
        "Look down"
    ]

    count = 0
    for inst in instructions:
        messagebox.showinfo("Capture", f"Please {inst} and press OK")
        ret, frame = cap.read()
        if ret:
            img_path = os.path.join(user_dir, f"{inst.replace(' ', '_')}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1

    cap.release()
    messagebox.showinfo("Success", f"Captured {count} images for {name}")

def verify():
    input_path = "live_test_data/input_image/input.jpg"
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    capture_image(input_path)

    input_img = preprocess(input_path)

    best_score = 0
    best_user = None

    for user in os.listdir(verification_dir):
        user_dir = os.path.join(verification_dir, user)
        results = []
        for file in os.listdir(user_dir):
            verification_img = preprocess(os.path.join(user_dir, file))
            result = model.predict([np.expand_dims(input_img, axis=0), 
                                    np.expand_dims(verification_img, axis=0)])
            results.append(result)

        score = np.mean(results)  # average similarity for that user
        if score > best_score:
            best_score = score
            best_user = user

    threshold = 0.5  # adjust as needed
    if best_score > threshold:
        messagebox.showinfo("Result", f"✅ Verified as {best_user} ({best_score*100:.2f}% match)")
    else:
        messagebox.showerror("Result", "❌ Access Denied")

# Build Tkinter GUI
root = tk.Tk()
root.title("Face Authentication")

add_btn = tk.Button(root, text="Add User", command=add_user, width=20, height=2)
add_btn.pack(pady=10)

verify_btn = tk.Button(root, text="Verify", command=verify, width=20, height=2)
verify_btn.pack(pady=10)

quit_btn = tk.Button(root, text="Quit", command=root.quit, width=20, height=2)
quit_btn.pack(pady=10)

root.mainloop()
