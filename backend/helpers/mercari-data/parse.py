import csv
import json
import sys
import os

def parse_tsv_to_json(tsv_file_path, json_file_path, limit=1000):
    print(f"Reading TSV file: {tsv_file_path}")
    
    try:
        with open(tsv_file_path, 'r', encoding='utf-8') as tsv_file:
            reader = csv.DictReader(tsv_file, delimiter='\t')
            
            data = []
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                data.append(row)
        
        print(f"Successfully read {len(data)} entries (limit: {limit})")
        
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted TSV to JSON. Saved to {json_file_path}")
        
        if data:
            print("\nPreview of first entry:")
            print(json.dumps(data[0], indent=2, ensure_ascii=False))
        
        return data
    
    except FileNotFoundError:
        print(f"Error: File {tsv_file_path} not found.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python tsv_to_json.py <input_tsv_file> <output_json_file> [limit]")
        print("  Example: python tsv_to_json.py test.tsv output.json 1000")
        
        default_input = 'train.tsv'
        default_output = 'mercari-set1.json'
        
        if os.path.exists(default_input):
            print(f"\nNo arguments provided. Using default example:")
            parse_tsv_to_json(default_input, default_output, 1000)
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    limit = 1000
    if len(sys.argv) > 3:
        try:
            limit = int(sys.argv[3])
        except ValueError:
            print(f"Error: Limit must be a number. Using default limit of 1000.")
    
    parse_tsv_to_json(input_file, output_file, limit)

def parse_tsv_to_json_custom(tsv_file_path, json_file_path, limit=1000):
    try:
        with open(tsv_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        headers = lines[0].strip().split('\t')
        
        data = []
        for i in range(1, min(len(lines), limit + 1)):
            line = lines[i].strip()
            
            values = []
            current_value = ""
            in_quotes = False
            j = 0
            
            while j < len(line):
                char = line[j]
                
                if char == '"':
                    if j+1 < len(line) and line[j+1] == '"':
                        current_value += '"'
                        j += 2
                        continue
                    else:
                        in_quotes = not in_quotes
                elif char == '\t' and not in_quotes:
                    values.append(current_value)
                    current_value = ""
                else:
                    current_value += char
                
                j += 1
            
            values.append(current_value)
            
            row_data = {}
            for j in range(len(headers)):
                if j < len(values):
                    value = values[j]
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    row_data[headers[j]] = value
                else:
                    row_data[headers[j]] = ""
            
            data.append(row_data)
        
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted TSV to JSON using custom parser. Saved to {json_file_path}")
        print(f"Total entries processed: {len(data)}")
        
        return data
    
    except Exception as e:
        print(f"Error in custom parser: {e}")
        return None

if __name__ == "__main__":
    main()