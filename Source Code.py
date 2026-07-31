# SIGNLEXICON v1.0
# AI Powered Sign Language Recognition System

# IMPORTS
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image

# LOAD CNN MODEL
MODEL_PATH = "05_Models/SignLearn_CNN.keras"
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully")

# CLASS LABELS
class_names = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z',
    'del','nothing','space'
]

# IMAGE PREDICTION FUNCTION
def upload_image():

    file_path = filedialog.askopenfilename(

        title="Choose an Image",

        filetypes=[
            ("Image Files","*.jpg *.jpeg *.png")
        ]
    )

    if file_path == "":
        return

    # SHOW IMAGE
    
    img = Image.open(file_path)
    
    img = img.resize((350,350))

    photo = ImageTk.PhotoImage(img)

    image_canvas.delete("all")

    image_canvas.create_image(
        175,
        175,
        image=photo
    )

    image_canvas.image = photo

    # CNN Prediction
    
    test_img = image.load_img(
        file_path,
        target_size=(200,200)
    )

    img_array = image.img_to_array(test_img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    predicted_class = class_names[predicted_index]

    confidence = np.max(prediction)


    prediction_label.config(
        text=f"Prediction : {predicted_class}"
    )

    confidence_label.config(
        text=f"Confidence : {confidence*100:.2f}%"
    )

# WEBCAM FUNCTION
def open_webcam():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        messagebox.showerror(
            "Error",
            "Cannot open webcam."
        )

        return


    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame,1)

        h,w,_ = frame.shape

        box = 400

        x1 = w//2 - box//2
        y1 = h//2 - box//2

        x2 = x1 + box
        y2 = y1 + box

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (255,0,0),
            2
        )

        roi = frame[y1:y2,x1:x2]

        roi = roi[30:-30,30:-30]

        img = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (200,200)
        )

        img = cv2.convertScaleAbs(
            img,
            alpha=1.3,
            beta=25
        )

        img = img.astype(np.float32)

        img = np.expand_dims(
            img,
            axis=0
        )

        prediction = model.predict(
            img,
            verbose=0
        )

        predicted_index = np.argmax(prediction)

        predicted_class = class_names[predicted_index]

        confidence = np.max(prediction)

        prediction_label.config(
            text=f"Prediction : {predicted_class}"
        )

        confidence_label.config(
            text=f"Confidence : {confidence*100:.2f}%"
        )

        root.update()

        if confidence >= 0.40:
            text = predicted_class
        else:
            text = "No Sign"

        cv2.putText(
            frame,
            f"{text} ({confidence*100:.2f}%)",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

        cv2.imshow(
            "SignLexicon Webcam",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()

    cv2.destroyAllWindows()

# GUI WINDOW

root = tk.Tk()

root.title("SignLexicon v1.0")
root.geometry("1050x650")
root.configure(bg="#F4F7FC")
root.resizable(False, False)

# HEADER
title = tk.Label(
    root,
    text="SIGNLEXICON",
    font=("Segoe UI", 28, "bold"),
    bg="#F4F7FC",
    fg="#0F4C81"
)

title.pack(pady=(18,2))

subtitle = tk.Label(
    root,
    text="AI Powered Sign Language Recognition System",
    font=("Segoe UI",13),
    bg="#F4F7FC",
    fg="gray35"
)

subtitle.pack()

# MAIN CONTAINER 
main_container = tk.Frame(
    root,
    bg="#F4F7FC"
)

main_container.pack(
    fill="both",
    expand=True,
    padx=40,
    pady=30
)

# IMAGE PANEL
image_frame = tk.LabelFrame(
    main_container,
    text=" Image Preview ",
    font=("Segoe UI",12,"bold"),
    bg="white",
    fg="#0F4C81",
    bd=2,
    relief="groove"
)

image_frame.place(
    x=20,
    y=10,
    width=390,
    height=390
)


image_canvas = tk.Canvas(
    image_frame,
    width=350,
    height=350,
    bg="white",
    highlightthickness=0
)

image_canvas.pack(
    padx=15,
    pady=15
)

image_canvas.create_text(
    175,
    175,
    text="No Image Selected",
    font=("Segoe UI",14),
    fill="gray"
)

# CONTROL PANEL
control_frame = tk.LabelFrame(
    main_container,
    text=" Controls ",
    font=("Segoe UI",12,"bold"),
    bg="#F4F7FC",
    fg="#0F4C81",
    bd=2,
    relief="groove"
)

control_frame.place(
    x=500,
    y=10,
    width=430,
    height=390
)

# BUTTONS
upload_btn = tk.Button(
    control_frame,
    text="📁 Upload Image",
    command=upload_image,
    font=("Segoe UI",12,"bold"),
    bg="#1976D2",
    fg="white",
    width=22,
    height=2,
    cursor="hand2"
)

upload_btn.pack(
    pady=(30,15)
)


webcam_btn = tk.Button(
    control_frame,
    text="📷 Open Webcam",
    command=open_webcam,
    font=("Segoe UI",12,"bold"),
    bg="#26A69A",
    fg="white",
    width=22,
    height=2,
    cursor="hand2"
)

webcam_btn.pack(
    pady=(0,35)
)

# RESULT LABELS
prediction_label = tk.Label(
    control_frame,
    text="Prediction : -",
    font=("Segoe UI",17,"bold"),
    bg="#F4F7FC",
    fg="#0F4C81"
)

prediction_label.pack(
    anchor="w",
    padx=40
)


confidence_label = tk.Label(
    control_frame,
    text="Confidence : -",
    font=("Segoe UI",15),
    bg="#F4F7FC",
    fg="#2E7D32"
)

confidence_label.pack(
    anchor="w",
    padx=40,
    pady=(15,0)
)

# BOTTOM SECTION
bottom_frame = tk.Frame(
    root,
    bg="#F4F7FC"
)

bottom_frame.pack(
    side="bottom",
    fill="x",
    pady=(0,25)
)

exit_btn = tk.Button(
    bottom_frame,
    text="Exit Application",
    command=root.destroy,
    font=("Segoe UI",12,"bold"),
    bg="#D32F2F",
    fg="white",
    width=20,
    height=2,
    cursor="hand2",
    activebackground="#B71C1C",
    activeforeground="white"
)

exit_btn.pack()

# FOOTER
footer = tk.Label(
    root,
    text="SignLexicon v1.0 © 2026",
    font=("Segoe UI",10),
    bg="#F4F7FC",
    fg="gray55"
)

footer.pack(side="bottom", pady=(0,10))

root.mainloop()