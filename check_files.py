import os

# Base directory (change this if needed)
base_path = os.path.join(os.getcwd(), 'data', 'raw')

# Files we expect
required_files = [
    'train.csv',
    'oil.csv',
    'holidays_events.csv',
    'stores.csv'
]

print(f"🔍 Checking files in: {base_path}\n")

# Check each file
for file in required_files:
    file_path = os.path.join(base_path, file)
    if os.path.exists(file_path):
        print(f"✅ Found: {file}")
    else:
        print(f"❌ Missing: {file}")

