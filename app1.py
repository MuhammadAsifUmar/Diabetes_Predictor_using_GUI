import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib
import pandas as pd
import os
import warnings

# Load the trained model and scaler
model = joblib.load(r'C:\Users\ASIF UMAR\Documents\GitHub\As_project\diabetes_model.pkl')
scaler = joblib.load(r'C:\Users\ASIF UMAR\Documents\GitHub\As_project\scaler.pkl')

# Ensure the data directory exists
data_directory = r'C:\Users\ASIF UMAR\Documents\GitHub\As_project\data'
os.makedirs(data_directory, exist_ok=True)

# Function to save input data to an Excel file
def save_data(name, input_data, result):
    file_path = os.path.join(data_directory, 'patient_data.xlsx')
    df = pd.DataFrame([input_data + [result]], columns=['Name of the Patient', 'Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age', 'Result'])
    if not os.path.exists(file_path):
        df.to_excel(file_path, index=False)
    else:
        df_existing = pd.read_excel(file_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_excel(file_path, index=False)

# Function to make predictions
def predict_diabetes():
    try:
        # Get user input
        name = name_entry.get()
        input_data = [name] + [float(entry.get()) for entry in entries]
        numeric_data = np.array(input_data[1:]).reshape(1, -1)

        # Handle warning for feature names
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            input_data_scaled = scaler.transform(numeric_data)

        # Make prediction
        prediction = model.predict(input_data_scaled)
        result = 'Diabetic' if prediction[0] == 1 else 'Not Diabetic'

        # Save input data
        save_data(name, input_data, result)

        # Display the result
        messagebox.showinfo("Prediction Result", f"The patient is: {result}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers.")

# Function to reset the input fields
def reset_fields():
    name_entry.delete(0, tk.END)
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
    ('Name of the Patient', ''),
    ('Pregnancies', ''),
    ('Glucose', ''),
    ('Blood Pressure (mm Hg)', ''),
    ('Skin Thickness (mm)', ''),
    ('Insulin (mu U/ml)', ''),
    ('Body Mass Index(BMI)', ''),
    ('Diabetes Pedigree Function', ''),
    ('Age', '')
]

entries = []

# Create the entry fields
name_label = tk.Label(root, text="Name", bg='lightgreen')
name_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
name_entry = tk.Entry(root)
name_entry.grid(row=1, column=1, padx=10, pady=5)

for i, (field, unit) in enumerate(fields[1:]):
    label = tk.Label(root, text=field, bg='lightgreen')
    label.grid(row=i+2, column=0, sticky=tk.W, padx=10, pady=5)
    
    entry = tk.Entry(root)
    entry.grid(row=i+2, column=1, padx=10, pady=5)
    entries.append(entry)

# Add predict button
predict_button = tk.Button(root, text="Predict", command=predict_diabetes, bg='white')
predict_button.grid(row=len(fields)+2, column=0, pady=20)

# Add reset button
reset_button = tk.Button(root, text="Reset", command=reset_fields, bg='white')
reset_button.grid(row=len(fields)+2, column=1, pady=20)

# Start the GUI event loop
root.mainloop()
