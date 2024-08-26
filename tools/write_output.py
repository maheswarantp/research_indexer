import os

if not os.path.exists("output"):
    os.makedirs("output")

def write_output_file(data):
    with open(f"output/output.txt", "w") as f:
        f.write(data)
    