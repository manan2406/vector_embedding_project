# generate_data.py
import pandas as pd
import random

# Lists of fabrics and colors for variety
fabrics = [
    "cotton", "silk", "wool", "linen", "polyester", "denim", "velvet", "nylon", "leather", "suede",
    "cashmere", "flannel", "chiffon", "satin", "tweed", "corduroy", "rayon", "bamboo", "hemp", "jute",
    "acrylic", "spandex", "modal", "viscose", "lace", "organza", "tulle", "muslin", "poplin", "gabardine"
]

colors = [
    "blue", "red", "green", "yellow", "black", "white", "purple", "orange", "pink", "grey",
    "brown", "beige", "navy", "teal", "maroon", "olive", "cyan", "magenta", "violet", "indigo",
    "gold", "silver", "coral", "turquoise", "lavender", "cream", "charcoal", "peach", "mint", "ruby"
]

# Descriptions for code_description
descriptions = [
    "for summer shirts", "for evening gowns", "for winter coats", "for casual wear", "for sportswear",
    "for formal shirts", "for luxury scarves", "for jeans", "for formal dresses", "for outdoor gear",
    "for sweaters", "for jackets", "for skirts", "for blouses", "for trousers", "for upholstery",
    "for curtains", "for bedding", "for bags", "for shoes", "for hats", "for gloves", "for ties",
    "for swimwear", "for activewear", "for sleepwear", "for party dresses", "for raincoats",
    "for vests", "for capes"
]

# Function to generate raw_material_code
def generate_code(index):
    return f"FAB{str(index+1).zfill(3)}"

# Generate combined data (200 rows total)
combined_data = {"product": [], "color": [], "raw_material_code": [], "code_description": []}
used_fabrics = set()

# First 100 rows (similar to sheet1 in the original)
for i in range(100):
    fabric = random.choice(fabrics)
    used_fabrics.add(fabric)
    color = random.choice(colors)
    code = generate_code(i)
    desc = f"{color.capitalize()} {fabric} {random.choice(descriptions)}"
    combined_data["product"].append(fabric)
    combined_data["color"].append(color)
    combined_data["raw_material_code"].append(code)
    combined_data["code_description"].append(desc)

# Second 100 rows (similar to sheet2 in the original)
for i in range(100):
    if i < 20 and used_fabrics:  # First 20 rows overlap with previously used fabrics
        fabric = random.choice(list(used_fabrics))
    else:
        fabric = random.choice(fabrics)
    used_fabrics.add(fabric)
    # Avoid exact duplicates by excluding colors used for this fabric in the first 100 rows
    if fabric in combined_data["product"]:
        used_colors = [combined_data["color"][j] for j, prod in enumerate(combined_data["product"]) if prod == fabric]
        available_colors = [c for c in colors if c not in used_colors]
        color = random.choice(available_colors) if available_colors else random.choice(colors)  # Fallback
    else:
        color = random.choice(colors)
    code = generate_code(i + 100)
    desc = f"{color.capitalize()} {fabric} {random.choice(descriptions)}"
    combined_data["product"].append(fabric)
    combined_data["color"].append(color)
    combined_data["raw_material_code"].append(code)
    combined_data["code_description"].append(desc)

# Create DataFrame and save to a single CSV
combined_df = pd.DataFrame(combined_data)
combined_df.to_csv("fabric_data.csv", index=False)
print("Single CSV file 'fabric_data.csv' has been created with 200 rows.")