import pypdf

pdf_path = "Final_paper_folder/Resource_Allocation_and_Robust_Trajectory_Design_for_Dual_UAV-Assisted_Secure_Communications.pdf"
reader = pypdf.PdfReader(pdf_path)

keywords = ["simulation", "noise", "power", "sensing", "table", "parameters", "sigma", "dbm"]

for page_num, page in enumerate(reader.pages):
    text = page.extract_text()
    if "simulation" in text.lower() or "table" in text.lower():
        print(f"=== PAGE {page_num + 1} ===")
        lines = text.split("\n")
        for line in lines:
            if any(kw.lower() in line.lower() for kw in keywords):
                print(f"  {line.strip()}")
