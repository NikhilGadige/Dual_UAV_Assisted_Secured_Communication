import pypdf
import sys

pdf_path = "Final_paper_folder/Resource_Allocation_and_Robust_Trajectory_Design_for_Dual_UAV-Assisted_Secure_Communications.pdf"
reader = pypdf.PdfReader(pdf_path)

keywords = ["uncertain", "eaves", "eve", "radius", "imperfect", "estimate", "location"]

print(f"Total pages: {len(reader.pages)}")

for page_num, page in enumerate(reader.pages):
    text = page.extract_text()
    matched = []
    for kw in keywords:
        if kw.lower() in text.lower():
            matched.append(kw)
    if matched:
        print(f"Page {page_num + 1} matches: {matched}")
        # Print a few lines containing the keywords
        lines = text.split("\n")
        for line in lines:
            if any(kw.lower() in line.lower() for kw in keywords):
                print(f"  [L] {line.strip()}")
