    import os
    import glob
    import asyncio
    from dotenv import load_dotenv
    from llama_parse import LlamaParse
    import yaml

    load_dotenv()

    async def parse_single_pdf(parser, pdf_path, output_dir):
        file_name = os.path.basename(pdf_path)
        base_name = os.path.splitext(file_name)[0]
        output_md_path = os.path.join(output_dir, f"{base_name}.md")

        print(f"Starting execution for: '{file_name}'...")
        try:
            documents = await parser.aload_data(pdf_path)
            
            # Injects ALL document metadata into the markdown file using YAML
            full_markdown = "\n\n".join([f"---\n{yaml.dump(doc.metadata)}---\n\n{doc.get_content()}" for doc in documents if doc.get_content()])

            if not full_markdown.strip():
                print(f"⚠️ Warning: LlamaParse returned empty text for '{file_name}'.")
                return False

            with open(output_md_path, "w", encoding="utf-8") as f:
                f.write(full_markdown)

            print(f"✅ Success! Saved {len(full_markdown)} characters to '{output_md_path}'")
            return True

        except Exception as e:
            print(f"❌ Error parsing {file_name}: {e}")
            return False

    async def main():
        llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not llama_key:
            print("Error: LLAMA_CLOUD_API_KEY not found in .env file.")
            return

        print("Initializing LlamaParse...")
        parser = LlamaParse(
            api_key=llama_key,
            result_type="markdown",
            verbose=True
        )

        raw_files = glob.glob("DATA/*.pdf") + glob.glob("DATA/*.PDF")
        pdf_files = sorted(list({os.path.abspath(f) for f in raw_files}))

        if not pdf_files:
            print("No PDF files found in 'DATA/' directory!")
            return

        output_dir = "parsed_data"
        os.makedirs(output_dir, exist_ok=True)

        # CRITICAL CHANGE: Creates individual tasks for all PDFs
        tasks = [parse_single_pdf(parser, pdf_path, output_dir) for pdf_path in pdf_files]
        
        print(f"Triggering parallel parsing for {len(tasks)} files. Please wait...\n")
        # Executes all processing tasks concurrently
        await asyncio.gather(*tasks)
        print("\n🎉 All files processed successfully!")

    if __name__ == "__main__":
        asyncio.run(main())
