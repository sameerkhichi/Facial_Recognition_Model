import cv2
import sys
import os
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from model import L1Dist
from util import preprocess

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller stores data files in a temp folder _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Load Siamese model
#model_path = resource_path("siamesemodelv2.h5")
model_path = resource_path("siamesemodelv2_keras")
model = tf.keras.models.load_model(
    model_path,
    custom_objects={"L1Dist": L1Dist, "BinaryCrossentropy": tf.losses.BinaryCrossentropy}
)

verification_dir = "app_data/verification_images"
os.makedirs(verification_dir, exist_ok=True)

#Global webcam
cap = cv2.VideoCapture(0)
current_frame = None
showing_feed = True #tracking if feed should be shown

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

def toggle_feed():
    global showing_feed
    showing_feed = not showing_feed
    toggle_btn.config(text="Hide Camera" if showing_feed else "Show Camera")

def capture_image(path):
    if current_frame is not None:
        cv2.imwrite(path, current_frame)

def add_user():
    name = simpledialog.askstring("Input", "Enter user name:")
    if not name:
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
            for i in range(10):  #burst of 10 images
                img_path = os.path.join(user_dir, f"{instructions[step].replace(' ', '_')}_{i}.jpg")
                capture_image(img_path)
            count += 1
            step += 1

            if step < len(instructions):
                instruction_label.config(text=f"{instructions[step]}")
            else:
                root.unbind("<Key>")
                instruction_label.config(text="Done capturing images")
                messagebox.showinfo("Success", f"Captured {count*10} images for {name}")
                #clear the label after 3 seconds
                root.after(3000, lambda: instruction_label.config(text=""))

    step = 0
    count = 0
    instruction_label.config(text=f"{instructions[step]}")
    root.bind("<Key>", key_handler)

#popup instead of messagebox which will fail with no console
def show_result_popup(message, title="Result"):
    """Creates a small popup window to display verification results."""
    popup = tk.Toplevel(root)
    popup.title(title)
    popup.configure(bg="#1e1e1e")
    popup.geometry("400x150")

    label = tk.Label(popup, text=message, font=("Helvetica", 16), fg="white", bg="#1e1e1e", wraplength=350)
    label.pack(pady=20)

    ok_btn = tk.Button(popup, text="OK", font=("Helvetica", 14), width=10, height=1, bg="#333", fg="white",
                       command=popup.destroy)
    ok_btn.pack(pady=10)

    # Make sure the popup appears above the main window
    popup.transient(root)
    popup.grab_set()
    root.wait_window(popup)

def verify():

    input_path = "app_data/input_image/input.jpg"
    os.makedirs(os.path.dirname(input_path), exist_ok=True)
    capture_image(input_path)

    input_img = preprocess(input_path)

    users = os.listdir(verification_dir)
    total_steps = sum(len(os.listdir(os.path.join(verification_dir, user))) for user in users)
    
    #Create a popup window for the progress bar
    progress_win = tk.Toplevel(root)
    progress_win.title("Verifying User...")
    progress_win.geometry("400x100")
    progress_win.configure(bg="#1e1e1e")
    progress_label = tk.Label(progress_win, text="Starting verification...", font=("Helvetica", 14), fg="white", bg="#1e1e1e")
    progress_label.pack(pady=10)
    progress_bar = ttk.Progressbar(progress_win, orient="horizontal", length=350, mode="determinate", maximum=total_steps)
    progress_bar.pack(pady=10)
    
    best_score = 0
    best_user = None
    step_count = 0

    for user in users:
        user_dir = os.path.join(verification_dir, user)
        results = []
        for file in os.listdir(user_dir):
            verification_img = preprocess(os.path.join(user_dir, file))
            result = model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(verification_img, axis=0)])
            results.append(result)

            # Update progress
            step_count += 1
            progress_bar['value'] = step_count
            progress_label.config(text=f"Verifying: {step_count}/{total_steps}")
            progress_win.update_idletasks()

        score = np.mean(results)
        if score > best_score:
            best_score = score
            best_user = user

    progress_label.config(text="Verification Complete")
    progress_bar['value'] = total_steps
    progress_win.update_idletasks()

    #Close progress popup after a short delay
    progress_win.after(500, progress_win.destroy)
   
    threshold = 0.3
    if best_score > threshold:
        show_result_popup(f"Verified as {best_user} ({best_score*100:.2f}% match)")
    else:
        show_result_popup("Access Denied")


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

toggle_btn = tk.Button(root, text="Hide Camera", command=toggle_feed, **btn_style)
toggle_btn.pack(pady=10)

quit_btn = tk.Button(root, text="Quit", command=root.quit, **btn_style)
quit_btn.pack(pady=10)

# Start webcam loop
show_frame()

root.mainloop()
cap.release()
cv2.destroyAllWindows()