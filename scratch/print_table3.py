import pypdf

pdf_path = "Final_paper_folder/Resource_Allocation_and_Robust_Trajectory_Design_for_Dual_UAV-Assisted_Secure_Communications.pdf"
reader = pypdf.PdfReader(pdf_path)
page = reader.pages[9]  # Page 10 is index 9
text = page.extract_text()

lines = text.split("\n")
for i, line in enumerate(lines):
    if "table iii" in line.lower() or "table ii" in line.lower():
        # Print next 25 lines
        for j in range(max(0, i-2), min(len(lines), i+25)):
            print(f"[{j}]: {lines[j]}")
        break
