import json

# Read lines from file
with open("fixed.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Convert to dictionary
data = {str(i + 1): line.strip() for i, line in enumerate(lines)}

# Write to JSON
with open("fixed.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Converted to fixed.json successfully!")
