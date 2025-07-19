import json
import os


def compile_notebooks_to_plaintext(output_filename, sort_by_name=True):
    notebook_files = [f for f in os.listdir('.') if f.endswith('.ipynb')]
    if sort_by_name:
        notebook_files.sort()

    print(f"Found {len(notebook_files)} notebook(s) to compile.\n")

    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for filename in notebook_files:
            print(f"Processing: {filename}...")
            outfile.write(f"\n{'='*80}\n# NOTEBOOK: {filename}\n{'='*80}\n\n")

            with open(filename, 'r', encoding='utf-8') as infile:
                notebook = json.load(infile)
                code_sources = [
                    cell.get('source', [])
                    for cell in notebook.get('cells', [])
                    if cell.get('cell_type') == 'code'
                ]

                if not code_sources:
                    outfile.write("# No code cells found in this notebook.\n\n")
                    continue

                for i, source_lines in enumerate(code_sources, 1):
                    outfile.write(f"# --- Code Cell {i} ---\n{''.join(source_lines)}\n\n")

    print(f"\nSuccessfully compiled all notebooks into '{output_filename}'.")

if __name__ == "__main__":
    output_filename = os.path.join('tmp', 'notebooks.txt')

    output_directory = os.path.dirname(output_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    compile_notebooks_to_plaintext(output_filename)
