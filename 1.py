import os
import glob
import nest_asyncio
from dotenv import load_dotenv
from llama_parse import LlamaParse

# Apply nest_asyncio to handle async execution in standard Python scripts
nest_asyncio.apply()

# Load environment variables from .env file
load_dotenv()

def parse_rbi_reports():
    # 1. Verify API Key
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
    if not llama_key:
        print(" Error: LLAMA_CLOUD_API_KEY not found in .env file. Please check your .env setup.")
        return

    print(" Initializing LlamaParse with Markdown result type...")
    parser = LlamaParse(
        api_key=llama_key,
        result_type="markdown",  # Preserves multi-column structure and markdown tables
        verbose=True
    )

    # 2. Find all PDF files in the DATA directory
    pdf_files = glob.glob("DATA/*.pdf") + glob.glob("DATA/*.PDF")
    if not pdf_files:
        print(" No PDF files found in the 'DATA' directory!")
        return

    print(f" Found {len(pdf_files)} PDF report(s) in 'DATA/':")
    for pdf in pdf_files:
        print(f"   - {pdf}")

    # Output folder for parsed markdown content
    output_dir = "parsed_data"
    os.makedirs(output_dir, exist_ok=True)

    # 3. Parse each PDF report
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        base_name = os.path.splitext(file_name)[0]
        output_md_path = os.path.join(output_dir, f"{base_name}.md")

        print(f"\n Parsing '{file_name}' (this may take a short moment depending on file size)...")
        try:
            # Send file to LlamaParse API
            documents = parser.load_data(pdf_path)

            # Combine page contents into a single markdown string
            full_markdown = "\n\n".join([doc.get_content() for doc in documents])

            # Save extracted markdown file locally
            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(full_markdown)

            print(f" Success! Extracted markdown saved to: '{output_md_path}'")

        except Exception as e:
            print(f" Error parsing {file_name}: {e}")

if __name__ == "__main__":
    parse_rbi_reports()