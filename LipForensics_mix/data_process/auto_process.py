import os

scripts = [
    # preparation
    'prepare.py',
    'img_to_npy.py',
]

for script in scripts:
    os.system(f'python {script}')
