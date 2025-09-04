import cv2
import os
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from model import L1Dist
from data_preprocessing import preprocess

# Load Siamese model
model = tf.keras.models.load_model(
    "siamesemodelv2.h5",
    custom_objects={"L1Dist": L1Dist, "BinaryCrossentropy": tf.losses.BinaryCrossentropy}
)

verification_dir = "app_data/verification_images"
os.makedirs(verification_dir, exist_ok=True)

#Global webcam
cap = cv2.VideoCapture(0)
current_frame = None
showing_feed = False #tracking if feed should be shown

def show_frame():
    """ Continuously updates webcam feed ONLY if active """
    global current_frame
    if showing_feed:
        ret, frame = cap.read()
        if ret:
            current_frame = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((100, 100))  # Display feed at 100x100
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
    else:
        video_label.configure(image="", text="")  # Clear feed when not in use
    video_label.after(10, show_frame)
def capture_image(path):
    if current_frame is not None:
        cv2.imwrite(path, current_frame)

def add_user():

    global showing_feed
    showing_feed = True #enable webcam


    name = simpledialog.askstring("Input", "Enter user name:")
    if not name:
        showing_feed = False
        return

    user_dir = os.path.join(verification_dir, name)
    os.makedirs(user_dir, exist_ok=True)

    instructions = [
        "Look straight",
        "Tilt head left",
        "Tilt head right",
        "Look up",
        "Look down"
    ]

    messagebox.showinfo("Instructions", "Follow prompts.\nPress SPACEBAR to capture each pose.")

    def key_handler(event):
        nonlocal step, count
        if event.keysym == "space":  #SPACE pressed
            #Capture burst of images for verification
            for i in range(7):  #burst of 7 images
                img_path = os.path.join(user_dir, f"{instructions[step].replace(' ', '_')}_{i}.jpg")
                capture_image(img_path)
            count += 1
            step += 1

            if step < len(instructions):
                instruction_label.config(text=f"{instructions[step]}")
            else:
                root.unbind("<Key>")
                instruction_label.config(text="Done capturing images")
                messagebox.showinfo("Success", f"Captured {count*7} images for {name}")
                showing_feed = False

    step = 0
    count = 0
    instruction_label.config(text=f"{instructions[step]}")
    root.bind("<Key>", key_handler)

def verify():

    global showing_feed
    showing_feed = True


    input_path = "app_data/input_image/input.jpg"
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
            result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(verification_img, axis=0)])
            results.append(result)

        score = np.mean(results)
        if score > best_score:
            best_score = score
            best_user = user

    showing_feed = False
    threshold = 0.5
    if best_score > threshold:
        messagebox.showinfo("Result", f"Verified as {best_user} ({best_score*100:.2f}% match)")
    else:
        messagebox.showerror("Result", "Access Denied")

# ------------------ GUI ------------------ #
root = tk.Tk()
root.title("Face Authentication App")
root.configure(bg="#1e1e1e")  #background
root.geometry("800x800")

title_label = tk.Label(root, text="Face Authentication System", 
                       font=("Helvetica", 24, "bold"), 
                       fg="white", bg="#1e1e1e")
title_label.pack(pady=20)

#Webcam feed label
video_label = tk.Label(root, bg="#1e1e1e")
video_label.pack(pady=20)

#Instruction label
instruction_label = tk.Label(root, text="", font=("Helvetica", 18), fg="lightblue", bg="#1e1e1e")
instruction_label.pack(pady=10)

#Buttons size
btn_style = {"font": ("Helvetica", 18), "width": 20, "height": 2, "bg": "#333", "fg": "white"}

add_btn = tk.Button(root, text="Add User", command=add_user, **btn_style)
add_btn.pack(pady=10)

verify_btn = tk.Button(root, text="Verify User", command=verify, **btn_style)
verify_btn.pack(pady=10)

quit_btn = tk.Button(root, text="Quit", command=root.quit, **btn_style)
quit_btn.pack(pady=10)

# Start webcam loop
show_frame()

root.mainloop()
cap.release()
cv2.destroyAllWindows()
