with open('sept_15.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace graphicspath to include Plots_20thOct
content = content.replace(
    r'\graphicspath{{./}{../}{./Week9 work/}{./output_final/}}',
    r'\graphicspath{{./}{../}{./Week9 work/}{./output_final/}{./Plots_20thOct/}}'
)

# Append Section IX before Conclusion
appendix_tex = r"""
\section{Comprehensive Empirical Evaluation \& Analysis of the 11 Benchmark Plots (October 21 Addendum)}
To provide a complete, rigorous characterization of the proposed Semantic-Aware RIS-assisted ISAC network, this section details all eleven targeted operational dimensions. Each analysis characterizes the underlying physical/mathematical mechanics, parameter regimes, empirical observations, and engineering implications for 6G network deployment.

\subsection{Plot 1: Dynamic Vehicle Mobility Modeling}
This empirical evaluation characterizes the kinematic behavior of terrestrial legitimate nodes (Mobile User $U$ and vehicular scatterers) under continuous time-varying kinematics. Ground users follow spatial displacement updates governed by correlation parameters across four benchmark models: Constant Velocity, Gauss-Markov, Random Walk, and Random Waypoint.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{01_vehicle_mobility.png}
    \caption{2D dynamic vehicle mobility trajectories and spatial displacement tracking over time.}
    \label{fig:plot01_mobility}
\end{figure}
\textbf{Key Engineering Takeaways:} Under Random Walk and Random Waypoint models, localized Brownian displacement keeps the user within a compact operational radius ($\approx 120\text{ m}$), allowing the aerial RIS UAV to maintain sharp beam-focusing with low steering angle variation. Constant velocity trajectories induce sustained Doppler phase offsets and elevation angle degradation, demonstrating the necessity of agile multi-UAV relay repositioning.

\subsection{Plot 2: Aerial RIS Element Configuration \& Phase Alignment}
Evaluates the passive reflection power scaling of the $N_R = 64$ element metasurface deployed on the aerial platform $R$. The plot benchmarks ideal constructive phase co-phasing ($\theta_r = -\angle(h_{RU,r}^* h_{BR,r})$) against uniform random phase scattering $\theta_r \sim \mathcal{U}[0, 2\pi)$.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{02_ris_element.png}
    \caption{Secrecy rate comparison under optimal constructive phase alignment vs. uncoordinated random phase shifts.}
    \label{fig:plot02_ris}
\end{figure}
\textbf{Key Engineering Takeaways:} Coherent phase alignment delivers quadratic power scaling proportional to $N_R^2 = 4096$ ($\approx 36\text{ dB}$ power gain), boosting legitimate link SINR far above detection thresholds. Due to distinct spatial angles of arrival toward eavesdroppers, phases aligned constructively for user $U$ scatter destructively/incoherently at eavesdroppers, ensuring strong secrecy rate enhancement.

\subsection{Plot 3: Cooperative Jammer Transmit Power Optimization}
Analyzes directional artificial noise projection emitted by UAV Jammer $J$ with spatial null-space suppression (leakage factor $\eta_{\text{ns}} = 10^{-4}$) directed at legitimate receivers Target $T$ and User $U$, sweeping $P_J$ from $0.05\text{ W}$ to $1.0\text{ W}$.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{03_jammer_power.png}
    \caption{Legitimate user SINR, worst-case eavesdropper SINR, and secrecy rate scaling as a function of Jammer Transmit Power ($P_J$).}
    \label{fig:plot03_jammer}
\end{figure}
\textbf{Key Engineering Takeaways:} Eavesdropper SINR plummets monotonically from $+2\text{ dB}$ down to below $-16\text{ dB}$ as jamming power increases, effectively blinding adversarial wiretap channels. Legitimate receivers sustain near-constant SINR due to null-space beamforming. Secrecy capacity exhibits diminishing returns beyond $P_J = 0.6\text{ W}$, identifying the optimal energy-efficient operational regime.

\subsection{Plot 4: Semantic Rate vs. Compression Ratio ($M/K$)}
Illustrates the fundamental trade-off between semantic unit density and channel decoding fidelity. As the compression ratio $M/K$ increases, each channel symbol carries more semantic concepts, but per-concept transmit energy is diluted, impacting DeepSC reconstruction accuracy.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{04_semantic_compression.png}
    \caption{Semantic transmission throughput ($M\text{ suts/s}$) as a function of semantic compression ratio $M/K$.}
    \label{fig:plot04_compression}
\end{figure}
\textbf{Key Engineering Takeaways:} Semantic throughput initially surges as $M/K$ rises from $0.1$ to $0.7$ due to enhanced spectral compactness. Beyond $M/K \approx 0.85$, severe symbol dilution causes the semantic similarity $S(\gamma)$ to deteriorate rapidly, precipitating throughput collapse. The optimal design point sits at $M/K = 2/3$ ($M=10, K=15$).

\subsection{Plot 5: DeepSC Semantic Similarity and Word Accuracy vs. Channel SINR}
Presents empirical results from the trained DeepSC architecture operating over fading channels, benchmarking sentence-level cosine embedding similarity against raw bilingual word accuracy across varying signal-to-noise-and-interference ratios.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{05_semantic_similarity.png}
    \caption{DeepSC transformer validation: semantic similarity and word accuracy across channel SINR from $-20\text{ dB}$ to $+20\text{ dB}$.}
    \label{fig:plot05_similarity}
\end{figure}
\textbf{Key Engineering Takeaways:} In low-SNR regimes ($-10\text{ dB}$ to $0\text{ dB}$), conventional bit-level word accuracy drops steeply below $55\%$, while semantic similarity remains remarkably robust ($> 0.86$). Above $6\text{ dB}$ SINR, semantic similarity saturates near $0.95$, proving that additional transmit power yields negligible semantic gain once conceptual meaning is recovered.

\subsection{Plot 6: Multi-UAV 3D Spatial Trajectory Control}
Depicts the learned continuous trajectory dynamics of the aerial RIS platform $R$ and the jamming platform $J$ navigating in a 3D airspace bounded by altitude constraints ($30-100\text{ m}$), maximum velocity ($20\text{ m/s}$), and inter-UAV collision thresholds ($d_{\text{col}} = 10\text{ m}$).
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{06_uav_trajectory.png}
    \caption{Continuous 3D flight trajectories of the RIS UAV and Jammer UAV under collision avoidance and tracking constraints.}
    \label{fig:plot06_trajectory}
\end{figure}
\textbf{Key Engineering Takeaways:} The RIS UAV dynamically positions itself at the midpoint between the BS and the ground cluster to balance cascaded path loss and preserve line-of-sight elevation angles. UAV $J$ establishes an offset patrol baseline that maximizes radar cross-section reflection from Target $T$ while maintaining angular separation to jam eavesdroppers without self-interference.

\subsection{Plot 7: Average System Secrecy Rate (ASSR) Convergence}
Tracks the policy optimization trajectory for legitimate secrecy across 8,000 learning episodes. Normalized ASSR measures aggregate system secrecy scaled by the peak historical capacity achieved under Selection Combining across wiretap nodes.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{07_assr.png}
    \caption{Multi-algorithm convergence comparison of Normalized Average Semantic Secrecy Rate (ASSR) over 8,000 episodes.}
    \label{fig:plot07_assr}
\end{figure}
\textbf{Key Engineering Takeaways:} Multi-Agent PPO (MAPPO) achieves rapid, monotonic convergence, attaining an asymptotic ASSR of $0.8418$ within 3,500 episodes. Single-agent baselines (SASAC, SATD3PG) suffer from non-stationary environment oscillations and plateau around $0.70-0.76$, while Random Walk fails to exceed $0.23$.

\subsection{Plot 8: Radar Target Detection Probability ($P_d$) Convergence}
Demonstrates radar target acquisition vigilance across training episodes. Detection probability combines target tracking and eavesdropper surveillance branches under Neyman-Pearson Marcum-Q function evaluation ($P_{fa} = 10^{-4}$).
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{08_detection_probability.png}
    \caption{Radar detection probability ($P_d$) learning curves at UAV Jammer $J$ under constant false alarm rate ($P_{fa} = 10^{-4}$).}
    \label{fig:plot08_pd}
\end{figure}
\textbf{Key Engineering Takeaways:} Both MAPPO and MATD3PG surpass the critical $P_d \ge 0.90$ reliability threshold within 2,000 episodes, proving that continuous artificial noise emission can coexist with radar sensing when waveform orthogonalization and null-space spatial projection are rigorously applied.

\subsection{Plot 9: Eavesdropper Uncertainty Radius Sensitivity Analysis}
Investigates the robustness of the worst-case physical layer security formulation by sweeping the spatial uncertainty sphere radius $\Delta Q_e$ from $2\text{ m}$ to $30\text{ m}$ around nominal coordinates.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{09_eavesdropper_radius.png}
    \caption{Impact of bounded eavesdropper location uncertainty radius ($\Delta Q_e$) on total secrecy rate and adversarial leakage.}
    \label{fig:plot09_radius}
\end{figure}
\textbf{Key Engineering Takeaways:} Increasing $\Delta Q_e$ assumes adversaries can be located closer to legitimate transmitters and further from the jammer, resulting in higher potential leakage. However, cooperative aerial jamming and RIS passive nulling ensure the secrecy rate remains well above $20\text{ k suts/s}$ even at severe uncertainty ($\Delta Q_e = 30\text{ m}$).

\subsection{Plot 10: Multi-Objective Utility Convergence ($U = \lambda_1 \text{ASSR} + \lambda_2 P_d$)}
Represents the holistic network fitness function uniting normalized semantic confidentiality (ASSR) and radar sensing vigilance ($P_d$) under strict Cramér-Rao Bound (CRB) feasibility constraints.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{10_combined_utility.png}
    \caption{Rolling 100-episode multi-objective utility convergence ($U = 0.5\mathrm{ASSR} + 0.5 P_d$) across all five benchmark algorithms.}
    \label{fig:plot10_utility}
\end{figure}
\textbf{Key Engineering Takeaways:} MAPPO achieves the highest steady-state utility ($U = 0.8717$), followed by MATD3PG ($0.8585$), SATD3PG ($0.8210$), SASAC ($0.7416$), and Random Walk ($0.3337$). Centralized training with decentralized execution (CTDE) in MAPPO significantly curtails policy variance, ensuring smooth optimization.

\subsection{Plot 11: Pareto Tradeoff Frontier (Detection Probability vs. Secrecy Rate)}
Visualizes the dual-objective trade-off manifold generated from 8,000 logged operational episodes, revealing the Pareto boundary between communication confidentiality and radar sensing sharpness.
\begin{figure}[H]
    \centering
    \includegraphics[width=0.68\textwidth]{11_tradeoff_dp_vs_secrecy.png}
    \caption{Empirical Pareto frontier between radar detection probability ($P_d$) and Average Semantic Secrecy Rate (ASSR).}
    \label{fig:plot11_pareto}
\end{figure}
\textbf{Key Engineering Takeaways:} The optimal network operating point lies at $P_d \in [0.88, 0.92]$ and $\text{ASSR} \in [0.82, 0.86]$. In this knee region, neither metric is disproportionately degraded. Operators can dynamically adjust weighting factors $(\lambda_1, \lambda_2)$ to favor ultra-secure communications or ultra-sharp radar surveillance along this verified Pareto frontier.
"""

idx = content.find(r'\section{Conclusion}')
if idx != -1:
    extended_content = content[:idx] + appendix_tex + '\n\n' + content[idx:]
else:
    extended_content = content + '\n\n' + appendix_tex

with open('21st_oct.tex', 'w', encoding='utf-8') as f:
    f.write(extended_content)

with open('Plots_20thOct/21st_oct.tex', 'w', encoding='utf-8') as f:
    f.write(extended_content)

print('Updated 21st_oct.tex in root and Plots_20thOct successfully')
