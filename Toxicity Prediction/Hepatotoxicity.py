import tkinter as tk
from tkinter import messagebox
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests

# Load dataset
data = pd.read_csv("Dataset.csv")

# Define thresholds for descriptors (example values)
thresholds = {
    'MaxAbsEStateIndex': 10.0,
    'MaxEStateIndex': 10.0,
    'MolWt': 500.0,
    'MolLogP': 5.0,
    'NumHDonors': 5,
    'NumHAcceptors': 10,
    'TPSA': 140.0,
    'NumRotatableBonds': 10,
    'RingCount': 5,
}


# Function to retrieve SMILES notation from PubChem using drug name
def get_smiles(drug_name):
    response = requests.get(
        f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name.lower()}/property/CanonicalSMILES/json")
    if response.status_code == 200:
        return response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
    else:
        return None


# Function to calculate molecular descriptors from SMILES notation
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        descriptors = {
            'MaxAbsEStateIndex': Descriptors.MaxAbsEStateIndex(mol),
            'MaxEStateIndex': Descriptors.MaxEStateIndex(mol),
            'MolWt': Descriptors.MolWt(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'RingCount': Descriptors.RingCount(mol),
        }
        return pd.Series(descriptors)
    else:
        return pd.Series([None] * 9, index=thresholds.keys())


# Apply function to create new columns for descriptors
descriptor_columns = list(thresholds.keys())
data[descriptor_columns] = data['SMILES'].apply(lambda x: calculate_descriptors(x))

# Drop rows with missing values
data.dropna(inplace=True)

# Define features and target variable
X = data[descriptor_columns]
y = data['Hepatotoxicity']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set and evaluate accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")


# Function to predict hepatotoxicity
def predict_toxicity(drug_name=None, smiles=None):
    if drug_name:
        smiles = get_smiles(drug_name)
        if not smiles:
            return None, "Drug not found on PubChem."
    if smiles:
        descriptors = calculate_descriptors(smiles)
        if None in descriptors.values:
            return None, "Invalid SMILES notation."
        # Create a DataFrame with the appropriate feature names
        descriptor_df = pd.DataFrame([descriptors], columns=descriptor_columns)
        prediction = "Hepatotoxic" if clf.predict(descriptor_df)[0] == 1 else "Non-Hepatotoxic"
        return descriptors, prediction
    return None, "SMILES notation not found."


# GUI creation
def create_gui():
    # Create main window
    root = tk.Tk()
    root.title("Hepatotoxicity Prediction")

    # Set the size and position of the window
    root.geometry("500x400+100+100")  # Width x Height + X_position + Y_position

    # Input fields
    tk.Label(root, text="Enter drug name:").grid(row=0, column=0, padx=10, pady=5)
    drug_name_entry = tk.Entry(root, width=30)
    drug_name_entry.grid(row=0, column=1, padx=10, pady=5)

    tk.Label(root, text="Enter SMILES notation:").grid(row=1, column=0, padx=10, pady=5)
    smiles_entry = tk.Entry(root, width=30)
    smiles_entry.grid(row=1, column=1, padx=10, pady=5)

    # Result display
    result_label = tk.Label(root, text="", font=("Arial", 12))
    result_label.grid(row=2, columnspan=2, padx=10, pady=10)

    # Table display
    table_frame = tk.Frame(root)
    table_frame.grid(row=3, columnspan=2, padx=30, pady=30)

    # Prediction function
    def on_predict():
        drug_name = drug_name_entry.get().strip()
        smiles = smiles_entry.get().strip()

        if not drug_name and not smiles:
            messagebox.showerror("Input Error", "Please enter either a drug name or a SMILES notation.")
            return

        descriptors, prediction = predict_toxicity(drug_name=drug_name, smiles=smiles)

        if descriptors is None:
            result_label.config(text=prediction, fg="red")
        else:
            result_label.config(text=f"Prediction: {prediction}",
                                fg="green" if prediction == "Non-Hepatotoxic" else "red")

            # Clear previous table
            for widget in table_frame.winfo_children():
                widget.destroy()

            # Display the table with color coding
            for i, (descriptor, value) in enumerate(descriptors.items()):
                tk.Label(table_frame, text=f"{descriptor}:").grid(row=i, column=0, sticky="w", padx=5)
                color = "red" if value > thresholds[descriptor] else "green"
                tk.Label(table_frame, text=f"{value:.2f}", fg=color).grid(row=i, column=1, sticky="w", padx=5)

    # Reset function
    def on_reset():
        drug_name_entry.delete(0, tk.END)
        smiles_entry.delete(0, tk.END)
        result_label.config(text="")
        for widget in table_frame.winfo_children():
            widget.destroy()

    # Buttons
    tk.Button(root, text="Predict", command=on_predict).grid(row=4, column=0, padx=10, pady=10)
    tk.Button(root, text="Reset", command=on_reset).grid(row=4, column=1, padx=10, pady=10)

    # Run the GUI loop
    root.mainloop()


if __name__ == "__main__":
    create_gui()
