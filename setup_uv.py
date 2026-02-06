import os
import sys
import platform

def main():
    # Define the target filename
    autodp_filename = "autodp.tar"
    cache_filename = "/uv/cache/"
    
    # Check if the file exists in the current directory
    if os.path.exists(autodp_filename):
        autodp_path = os.path.abspath(autodp_filename)
        print(f"Found {autodp_filename} at: {autodp_path}")
    else:
        # If not found, ask the user for the path
        print(f"{autodp_filename} not found in the current directory.")
        user_input = input(f"Please enter the full path to {autodp_filename}: ").strip()
        
        # Remove quotes if the user added them (common when copying paths)
        if (user_input.startswith('"') and user_input.endswith('"')) or \
           (user_input.startswith("'") and user_input.endswith("'")):
            user_input = user_input[1:-1]
            
        if os.path.exists(user_input) and os.path.isfile(user_input):
            autodp_path = os.path.abspath(user_input)
            print(f"Verified file at: {autodp_path}")
        else:
            print(f"Error: File not found at {user_input}")
            sys.exit(1)

    # Read the example file
    example_file = "pyproject.toml.example"
    if not os.path.exists(example_file):
        print(f"Error: {example_file} not found.")
        sys.exit(1)

    with open(example_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Modify line 20 (index 19)
    # The requirement is: "autodp @ file:\\\E:\Project\dolphin_proto\autodp.tar"
    # Note: The user specified triple backslash for the file URI scheme which is a bit unusual for standard file URIs 
    # (usually file:///), but I will stick to their requested format: file:\\\Path
    
    # Ensure we have enough lines
    if len(lines) < 23:
        print(f"Error: {example_file} has fewer than 23 lines.")
        sys.exit(1)
    
    # Construct the new line
    # Adjust based on OS
    if platform.system() == "Windows":
        # Windows: ensure backslashes
        cache_filename = "uv\\cache\\"
        cache_path = os.path.abspath(".")[:3]+cache_filename
        formatted_autodp_path = autodp_path.replace("\\", "\\\\")
        formatted_cache_path = cache_path.replace("\\", "\\\\")
        auto_dp_content = f'    "autodp @ file://{formatted_autodp_path}",\n'
        cache_path_content = f'cache-dir = "{formatted_cache_path}"\n'
    else:
        cache_path = os.path.expanduser("~")+cache_filename
        formatted_autodp_path = autodp_path
        formatted_cache_path = cache_path
        #formatted_autodp_path = autodp_path.replace("\\", "\\\\")
        #formatted_cache_path = cache_path.replace("\\", "\\\\")
        auto_dp_content = f'    "autodp @ file://{formatted_autodp_path}",\n'
        cache_path_content = f'cache-dir = "{formatted_cache_path}"\n'
    
    # Check if the original line looks like what we expect to replace (sanity check)
    # Line 23 in example is: "    autodp @ file:autodp-0.1.1.tar,"
    # We will replace it regardless, but good to know.
    
    lines[1] = cache_path_content
    lines[22] = auto_dp_content
    
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
        print("Cache folder created")

    # Write to pyproject.toml
    output_file = "pyproject.toml"
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    print(f"Successfully created {output_file} with autodp path: {autodp_path}")

if __name__ == "__main__":
    main()
