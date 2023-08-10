import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox

# Suppress warnings
warnings.filterwarnings("ignore")

# 1. Importing the Dataset
data = pd.read_excel(r"C:\Users\ashit\Desktop\P\Original Train.xlsx")
data_test = pd.read_excel(r"C:\Users\ashit\Desktop\P\Original Test.xlsx")

# 2. Display Top 5 Rows of The Dataset
data.head()

# 3. Check Last 5 Rows of The Dataset
data.tail()

# 4. Find Shape of Our Dataset (Number of Rows And Number of Columns)
data_shape = data.shape
print("Number of Rows:", data_shape[0])
print("Number of Columns:", data_shape[1])

# 5. Taking Care of Duplicate Values
duplicated_columns = data.columns[data.T.duplicated()].tolist()
len_duplicated_columns = len(duplicated_columns)

# 6. Taking Care of Missing Values
missing_values = data.isnull().sum()

# Visualize Activity distribution
activity_counts = data['Activity'].value_counts()
activity_counts.plot(kind='bar')
plt.xticks(rotation=35)
plt.show()


# 7. Store Feature Matrix In X and Response(Target) In Vector y
X = data.drop('Activity', axis=1)
y = data['Activity']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 8. Splitting The Dataset Into The Training Set And Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=42)

# 9. Logistic Regression
log = LogisticRegression()
log.fit(X_train, y_train)
y_pred1 = log.predict(X_test)
accuracy_lr = accuracy_score(y_test, y_pred1)

# 10. Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred2 = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred2)

# Save Random Forest Classifier model
joblib.dump(rf, "model_rf.pkl")

# Define GUI functions
def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("Excel Files", "*.xlsx")])
    if filepath:
        try:
            data = pd.read_excel(filepath)
            process_data(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open file: {e}")

def process_data(data):
    duplicated_columns = data.columns[data.T.duplicated()].tolist()
    data_test = data.drop(duplicated_columns, axis=1)

    model = joblib.load("model_rf.pkl")
    y_pred = model.predict(data_test)
    y_pred = le.inverse_transform(y_pred)
    data['Predicted_Activity'] = y_pred
    save_file(data)

def save_file(data):
    savepath = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
    if savepath:
        try:
            data.to_excel(savepath, index=False)
            messagebox.showinfo("Success", "File Saved Successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file: {e}")

# Create a Tkinter GUI
root = tk.Tk()
root.title("Classification")
root.geometry("200x200")

button1 = tk.Button(root, text="Open Excel File", width=15, height=2,
                    background="lightgreen", activebackground="lightblue",
                    font=("Arial", 11, "bold"), command=open_file)
button1.pack(pady=50)

root.mainloop()
