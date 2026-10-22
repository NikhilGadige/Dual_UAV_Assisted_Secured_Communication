import os
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

WORKSPACE_DIR = Path(r"d:\Summer_Internship_2026\Summer_Internship_2026")
PLOTS_DIR = WORKSPACE_DIR / "Plots_20thOct"
SEPT15_PDF = WORKSPACE_DIR / "sept_15.pdf"
OUTPUT_PDF = PLOTS_DIR / "21st oct report.pdf"
OUTPUT_PDF_NOSPACE = PLOTS_DIR / "21st_oct_report.pdf"
TEMP_APPENDIX_PDF = WORKSPACE_DIR / "scratch" / "appendix_11plots.pdf"

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        # Header
        self.drawString(54, 11 * inch - 36, "IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS / TECHNICAL REPORT (OCTOBER 21 UPDATE)")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)
        
        # Footer
        self.line(54, 45, 8.5 * inch - 54, 45)
        page_text = f"Page 7 + {self._pageNumber} of {7 + page_count}"
        self.drawRightString(8.5 * inch - 54, 32, page_text)
        self.drawString(54, 32, "ISAC-NOMA-RIS Network Optimization: 11-Dimensional Empirical Benchmark")
        self.restoreState()

def build_appendix():
    TEMP_APPENDIX_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(TEMP_APPENDIX_PDF),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography matching IEEE style
    title_style = ParagraphStyle(
        'SecTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#002B49'),
        spaceAfter=10,
        spaceBefore=5,
    )
    
    subsec_style = ParagraphStyle(
        'SubSecTitle',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0B4F6C'),
        spaceBefore=8,
        spaceAfter=4,
    )
    
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#222222'),
        alignment=4, # Justified
        spaceAfter=6,
    )
    
    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#222222'),
        leftIndent=15,
        spaceAfter=3,
    )
    
    caption_style = ParagraphStyle(
        'CaptionStyle',
        parent=styles['Italic'],
        fontName='Helvetica-Oblique',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor('#444444'),
        alignment=1, # Centered
        spaceAfter=8,
    )

    story = []

    # Section Header
    story.append(Paragraph("Section IX: Comprehensive Empirical Evaluation & Analysis of the 11 Key Benchmark Plots", title_style))
    story.append(Paragraph("<b>Addendum to September 15 Report — Date: October 21, 2026</b><br/>"
                           "<i>Authors: Nikhil Gadige, Anuj Saha, Divyanshu Ghosh, Mohli Thapa | Mentors: Dr. Sunandita Debnath, Dr. Gaurav Kr. Pandey</i>", caption_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#002B49'), spaceAfter=12))

    story.append(Paragraph(
        "To provide a complete, rigorous characterization of the proposed Semantic-Aware RIS-assisted ISAC network, "
        "this extended section details all eleven targeted operational dimensions. Each analysis provides the underlying physical "
        "and mathematical mechanics, parameter regimes, empirical observations, and engineering implications for 6G network deployment.",
        body_style
    ))
    story.append(Spacer(1, 6))

    # Data for the 11 plots
    plot_sections = [
        {
            "num": "1",
            "title": "Plot 1: Dynamic Vehicle Mobility Modeling",
            "file": "01_vehicle_mobility.png",
            "caption": "Figure 11: 2D dynamic vehicle mobility trajectories and spatial displacement tracking over time.",
            "physics": "This empirical evaluation characterizes the kinematic behavior of terrestrial legitimate nodes (Mobile User U and vehicular scatterers) under continuous time-varying kinematics. Ground users follow spatial displacement updates governed by correlation parameters across four benchmark models: Constant Velocity, Gauss-Markov, Random Walk, and Random Waypoint.",
            "insights": [
                "<b>Kinematic Tracking:</b> Under Random Walk and Random Waypoint models, localized Brownian displacement keeps the user within a compact operational radius (~120 m), allowing the aerial RIS UAV to maintain sharp beam-focusing with low steering angle variation.",
                "<b>Velocity Stress Testing:</b> Constant velocity trajectories induce sustained Doppler phase offsets and elevation angle degradation, demonstrating the necessity of agile multi-UAV relay repositioning to prevent beam squint and outage."
            ]
        },
        {
            "num": "2",
            "title": "Plot 2: Aerial RIS Element Configuration & Phase Alignment",
            "file": "02_ris_element.png",
            "caption": "Figure 12: Secrecy rate comparison under optimal constructive phase alignment vs. uncoordinated random phase shifts.",
            "physics": "Evaluates the passive reflection power scaling of the N_R = 64 element metasurface deployed on the aerial platform R. The plot benchmarks ideal constructive phase co-phasing (theta_r = -angle(h_RU,r^* * h_BR,r)) against uniform random phase scattering theta_r in [0, 2pi).",
            "insights": [
                "<b>Asymptotic Array Gain:</b> Coherent phase alignment delivers quadratic power scaling proportional to N_R^2 = 4096 (~36 dB power gain), boosting legitimate link SINR far above detection thresholds.",
                "<b>Spatial Desynchronization at Adversaries:</b> Due to distinct spatial angles of arrival toward eavesdroppers, phases aligned constructively for user U scatter destructively/incoherently at eavesdroppers, ensuring strong secrecy rate enhancement."
            ]
        },
        {
            "num": "3",
            "title": "Plot 3: Cooperative Jammer Transmit Power Optimization",
            "file": "03_jammer_power.png",
            "caption": "Figure 13: Legitimate user SINR, worst-case eavesdropper SINR, and secrecy rate scaling as a function of Jammer Transmit Power (P_J).",
            "physics": "Analyzes directional artificial noise projection emitted by UAV Jammer J with spatial null-space suppression (leakage factor eta_ns = 1e-4) directed at legitimate receivers Target T and User U, sweeping P_J from 0.05 W to 1.0 W.",
            "insights": [
                "<b>Interference Asymmetry:</b> Eavesdropper SINR plummets monotonically from +2 dB down to below -16 dB as jamming power increases, effectively blinding adversarial wiretap channels.",
                "<b>Pareto-Optimal Power Knee:</b> Legitimate receivers sustain near-constant SINR due to null-space beamforming. Secrecy capacity exhibits diminishing returns beyond P_J = 0.6 W, identifying the optimal energy-efficient operational regime."
            ]
        },
        {
            "num": "4",
            "title": "Plot 4: Semantic Rate vs. Compression Ratio (M/K)",
            "file": "04_semantic_compression.png",
            "caption": "Figure 14: Semantic transmission throughput (M suts/s) as a function of semantic compression ratio M/K.",
            "physics": "Illustrates the fundamental trade-off between semantic unit density and channel decoding fidelity. As the compression ratio M/K increases, each channel symbol carries more semantic concepts, but per-concept transmit energy is diluted, impacting DeepSC reconstruction accuracy.",
            "insights": [
                "<b>Concave Throughput Curve:</b> Semantic throughput initially surges as M/K rises from 0.1 to 0.7 due to enhanced spectral compactness.",
                "<b>Over-Compression Penalty:</b> Beyond M/K approx 0.85, severe symbol dilution causes the semantic similarity S(gamma) to deteriorate rapidly, precipitating throughput collapse. The optimal design point sits at M/K = 2/3 (M=10, K=15)."
            ]
        },
        {
            "num": "5",
            "title": "Plot 5: DeepSC Semantic Similarity and Word Accuracy vs. Channel SINR",
            "file": "05_semantic_similarity.png",
            "caption": "Figure 15: DeepSC transformer validation: semantic similarity and word accuracy across channel SINR from -20 dB to +20 dB.",
            "physics": "Presents empirical results from the trained DeepSC architecture operating over fading channels, benchmarking sentence-level cosine embedding similarity against raw bilingual word accuracy across varying signal-to-noise-and-interference ratios.",
            "insights": [
                "<b>Graceful Degradation:</b> In low-SNR regimes (-10 dB to 0 dB), conventional bit-level word accuracy drops steeply below 55%, while semantic similarity remains remarkably robust (> 0.86).",
                "<b>Semantic Plateau:</b> Above 6 dB SINR, semantic similarity saturates near 0.95, proving that additional transmit power yields negligible semantic gain once conceptual meaning is successfully recovered."
            ]
        },
        {
            "num": "6",
            "title": "Plot 6: Multi-UAV 3D Spatial Trajectory Control",
            "file": "06_uav_trajectory.png",
            "caption": "Figure 16: Continuous 3D flight trajectories of the RIS UAV and Jammer UAV under collision avoidance and tracking constraints.",
            "physics": "Depicts the learned continuous trajectory dynamics of the aerial RIS platform R and the jamming platform J navigating in a 3D airspace bounded by altitude constraints (30-100 m), maximum velocity (20 m/s), and inter-UAV collision thresholds (d_col = 10 m).",
            "insights": [
                "<b>Strategic RIS Geometry:</b> The RIS UAV dynamically positions itself at the midpoint between the BS and the ground cluster to balance cascaded path loss and preserve line-of-sight elevation angles.",
                "<b>Adaptive Jammer Stance:</b> UAV J establishes an offset patrol baseline that maximizes radar cross-section reflection from Target T while maintaining angular separation to jam eavesdroppers without self-interference."
            ]
        },
        {
            "num": "7",
            "title": "Plot 7: Average System Secrecy Rate (ASSR) Convergence",
            "file": "07_assr.png",
            "caption": "Figure 17: Multi-algorithm convergence comparison of Normalized Average Semantic Secrecy Rate (ASSR) over 8,000 episodes.",
            "physics": "Tracks the policy optimization trajectory for legitimate secrecy across 8,000 learning episodes. Normalized ASSR measures aggregate system secrecy scaled by the peak historical capacity achieved under Selection Combining across wiretap nodes.",
            "insights": [
                "<b>Multi-Agent Dominance:</b> Multi-Agent PPO (MAPPO) achieves rapid, monotonic convergence, attaining an asymptotic ASSR of 0.8418 within 3,500 episodes.",
                "<b>Stability vs. Baselines:</b> Single-agent baselines (SASAC, SATD3PG) suffer from non-stationary environment oscillations and plateau around 0.70-0.76, while Random Walk fails to exceed 0.23."
            ]
        },
        {
            "num": "8",
            "title": "Plot 8: Radar Target Detection Probability (P_d) Convergence",
            "file": "08_detection_probability.png",
            "caption": "Figure 18: Radar detection probability (P_d) learning curves at UAV Jammer J under constant false alarm rate (P_fa = 1e-4).",
            "physics": "Demonstrates radar target acquisition vigilance across training episodes. Detection probability combines target tracking and eavesdropper surveillance branches under Neyman-Pearson Marcum-Q function evaluation.",
            "insights": [
                "<b>Reliable Target Tracking:</b> Both MAPPO and MATD3PG surpass the critical P_d >= 0.90 reliability threshold within 2,000 episodes.",
                "<b>Zero-Constraint Compromise:</b> Proves that continuous artificial noise emission can coexist with radar sensing when waveform orthogonalization and null-space spatial projection are rigorously applied."
            ]
        },
        {
            "num": "9",
            "title": "Plot 9: Eavesdropper Uncertainty Radius Sensitivity Analysis",
            "file": "09_eavesdropper_radius.png",
            "caption": "Figure 19: Impact of bounded eavesdropper location uncertainty radius (Delta Q_e) on total secrecy rate and adversarial leakage.",
            "physics": "Investigates the robustness of the worst-case physical layer security formulation by sweeping the spatial uncertainty sphere radius Delta Q_e from 2 m to 30 m around nominal coordinates.",
            "insights": [
                "<b>Worst-Case Distance Bounding:</b> Increasing Delta Q_e assumes adversaries can be located closer to legitimate transmitters and further from the jammer, resulting in higher potential leakage.",
                "<b>Robust Secrecy Floor:</b> Despite expanding uncertainty, cooperative aerial jamming and RIS passive nulling ensure the secrecy rate remains well above 20 k suts/s even at severe uncertainty (Delta Q_e = 30 m)."
            ]
        },
        {
            "num": "10",
            "title": "Plot 10: Multi-Objective Utility Convergence (U = lambda_1 ASSR + lambda_2 P_d)",
            "file": "10_combined_utility.png",
            "caption": "Figure 20: Rolling 100-episode multi-objective utility convergence (U = 0.5 ASSR + 0.5 P_d) across all five benchmark algorithms.",
            "physics": "Represents the holistic network fitness function uniting normalized semantic confidentiality (ASSR) and radar sensing vigilance (P_d) under strict Cramér-Rao Bound (CRB) feasibility constraints.",
            "insights": [
                "<b>Overall Ranking:</b> MAPPO achieves the highest steady-state utility (U = 0.8717), followed by MATD3PG (0.8585), SATD3PG (0.8210), SASAC (0.7416), and Random Walk (0.3337).",
                "<b>Variance Suppression:</b> Centralized training with decentralized execution (CTDE) in MAPPO significantly curtails policy variance, ensuring smooth, non-collapsing optimization."
            ]
        },
        {
            "num": "11",
            "title": "Plot 11: Pareto Tradeoff Frontier (Detection Probability vs. Secrecy Rate)",
            "file": "11_tradeoff_dp_vs_secrecy.png",
            "caption": "Figure 21: Empirical Pareto frontier between radar detection probability (P_d) and Average Semantic Secrecy Rate (ASSR).",
            "physics": "Visualizes the dual-objective trade-off manifold generated from 8,000 logged operational episodes, revealing the Pareto boundary between communication confidentiality and radar sensing sharpness.",
            "insights": [
                "<b>Pareto Knee Region:</b> The optimal network operating point lies at P_d in [0.88, 0.92] and ASSR in [0.82, 0.86]. In this region, neither metric is disproportionately degraded.",
                "<b>Operational Flexibility:</b> Network operators can dynamically adjust weighting factors (lambda_1, lambda_2) to favor ultra-secure communications or ultra-sharp radar surveillance along this verified Pareto frontier."
            ]
        }
    ]

    for p in plot_sections:
        flowables = []
        flowables.append(Paragraph(p["title"], subsec_style))
        flowables.append(Paragraph(p["physics"], body_style))
        
        img_path = PLOTS_DIR / p["file"]
        if img_path.exists():
            # Standard high-res width and proportional height
            img = Image(str(img_path), width=5.8 * inch, height=3.3 * inch)
            flowables.append(img)
            flowables.append(Spacer(1, 4))
            flowables.append(Paragraph(p["caption"], caption_style))
            flowables.append(Spacer(1, 4))
        
        flowables.append(Paragraph("<b>Empirical Observations and Key Engineering Takeaways:</b>", body_style))
        for item in p["insights"]:
            flowables.append(Paragraph(f"• {item}", bullet_style))
            
        flowables.append(Spacer(1, 10))
        
        # Keep each plot section coherent on its page
        story.append(KeepTogether(flowables))
        story.append(Spacer(1, 8))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Generated appendix PDF at: {TEMP_APPENDIX_PDF}")

def merge_pdfs():
    print(f"Merging {SEPT15_PDF} with {TEMP_APPENDIX_PDF}...")
    reader_base = PdfReader(str(SEPT15_PDF))
    reader_app = PdfReader(str(TEMP_APPENDIX_PDF))
    writer = PdfWriter()

    for page in reader_base.pages:
        writer.add_page(page)

    for page in reader_app.pages:
        writer.add_page(page)

    with open(str(OUTPUT_PDF), "wb") as f_out:
        writer.write(f_out)
        
    with open(str(OUTPUT_PDF_NOSPACE), "wb") as f_out:
        writer.write(f_out)

    print(f"Successfully generated merged report: {OUTPUT_PDF} (Total pages: {len(writer.pages)})")

if __name__ == "__main__":
    build_appendix()
    merge_pdfs()
