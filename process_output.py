import json
import sys

def process_output(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)

            with open(output_file, 'w', encoding='utf-8') as output_file:
                for item in data:
                    output_file.write(item + '\n')

        except json.JSONDecodeError:
            print("Error decoding JSON in the file.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_output.py <input_file> <output_file>")
        sys.exit(1)

    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    process_output(input_file_path, output_file_path)
