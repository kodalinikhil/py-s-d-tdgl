import os
import glob

def replace_in_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Replacements list
    replacements = [
        ('psi_d', 'psi2'),
        ('psi_s', 'psi1'),
        ('alpha_d', 'alpha2'),
        ('alpha_s', 'alpha1'),
        ('beta_d', 'beta2'),
        ('beta_s', 'beta1'),
        ('gamma_d', 'eta2'),
        ('gamma_s', 'eta1'),
    ]

    new_content = content
    for old, new in replacements:
        new_content = new_content.replace(old, new)

    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

# Search for python files
for root, _, files in os.walk('tdgl'):
    for file in files:
        if file.endswith('.py'):
            replace_in_file(os.path.join(root, file))

for root, _, files in os.walk('my_scripts'):
    for file in files:
        if file.endswith('.py'):
            replace_in_file(os.path.join(root, file))
