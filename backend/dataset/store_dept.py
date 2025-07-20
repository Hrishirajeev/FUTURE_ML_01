import pandas as pd

# Update the path if your file is elsewhere
csv_path = "train.csv"

# Read the CSV
df = pd.read_csv(csv_path)

# Get unique store and department numbers as plain integers
unique_stores = sorted([int(x) for x in df['Store'].unique()])
unique_departments = sorted([int(x) for x in df['Dept'].unique()])

print("Unique Stores:")
print(unique_stores)
print("\nUnique Departments:")
print(unique_departments)

# If you want to see which departments are available for each store:
print("\nDepartments available for each store:")
for store in unique_stores:
    depts = sorted([int(x) for x in df[df['Store'] == store]['Dept'].unique()])
    print(f"Store {store}: {depts}")