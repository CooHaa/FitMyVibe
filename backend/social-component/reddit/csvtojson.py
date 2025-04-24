import csv
import json
import sys
import os

def parse_csv_to_json(csv_file_path, json_file_path, limit=1000):
    print(f"Reading CSV file: {csv_file_path}")
    try:
        with open(csv_file_path, 'r', encoding='utf-8', newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            data = []
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                data.append(row)
        print(f"Successfully read {len(data)} entries (limit: {limit})")

        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)
        print(f"Successfully converted CSV to JSON (first {len(data)} rows). Saved to {json_file_path}")

        if data:
            print("\nPreview of first entry:")
            print(json.dumps(data[0], indent=2, ensure_ascii=False))

        return data

    except FileNotFoundError:
        print(f"Error: File {csv_file_path} not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    if len(sys.argv) < 3:
        default_input = 'vestiaire.csv'
        default_output = 'vestiaire-data.json'
        default_limit = 500
        if os.path.exists(default_input):
            parse_csv_to_json(default_input, default_output, default_limit)
        return

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    limit = 1000
    if len(sys.argv) > 3:
        try:
            limit = int(sys.argv[3])
        except ValueError:
            print("f")

    parse_csv_to_json(input_file, output_file, limit)

if __name__ == "__main__":
    main()
