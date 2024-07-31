import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load(r'C:\Users\ASIF UMAR\Documents\GitHub\As_project\diabetes_model.pkl')
scaler = joblib.load(r'C:\Users\ASIF UMAR\Documents\GitHub\As_project\scaler.pkl')

# Function to make predictions
def predict_diabetes():
    try:
        # Get user input
        input_data = [float(entry.get()) for entry in entries]
        input_data = np.array(input_data).reshape(1, -1)
        input_data_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

        # Display the result
        messagebox.showinfo("Prediction Result", f"The patient is: {result}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")

# Function to reset the input fields
def reset_fields():
    for entry in entries:
        entry.delete(0, tk.END)

# Create the main window
root = tk.Tk()
root.title("Diabetes Prediction")
root.configure(bg='lightgreen')

title = tk.Label(root, text="The Diabetes Model Predictor Made By Muhammad Asif Umar (AI)", bg='black', fg='red', font=("Algerian", 16, "bold"))
title.grid(row=0, columnspan=2, pady=10)

# Define labels and entry fields
fields = [
    ('Pregnancies', ''),
    ('Glucose', ''),
    ('Blood Pressure (mm Hg)', ''),
    ('Skin Thickness (mm)', ''),
    ('Insulin (mu U/ml)', ''),
    ('BMI', ''),
    ('Diabetes Pedigree Function', ''),
    ('Age', '')
]

entries = []

for i, (field, unit) in enumerate(fields):
    label = tk.Label(root, text=field, bg='lightgreen')
    label.grid(row=i+1, column=0, sticky=tk.W, padx=10, pady=5)
    
    entry = tk.Entry(root)
    entry.grid(row=i+1, column=1, padx=10, pady=5)
    entries.append(entry)

# Add predict button
predict_button = tk.Button(root, text="Predict", command=predict_diabetes, bg='white')
predict_button.grid(row=len(fields)+1, column=0, pady=20)

# Add reset button
reset_button = tk.Button(root, text="Reset", command=reset_fields, bg='white')
reset_button.grid(row=len(fields)+1, column=1, pady=20)

# Start the GUI event loop
root.mainloop()
