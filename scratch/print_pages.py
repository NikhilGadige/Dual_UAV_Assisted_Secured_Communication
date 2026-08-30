import pypdf

pdf_path = "Final_paper_folder/Resource_Allocation_and_Robust_Trajectory_Design_for_Dual_UAV-Assisted_Secure_Communications.pdf"
reader = pypdf.PdfReader(pdf_path)

for p in [3, 4, 13]:
    print(f"=== PAGE {p} ===")
    print(reader.pages[p-1].extract_text())
    print("\n\n")
