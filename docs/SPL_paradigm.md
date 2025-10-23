# Stability Primitive Lattice (SPL) — A Paradigm for Universal Robot Policy Reliability

## 1. Research Vision and Target Impact
- **Goal**: establish *Stability Primitive Lattice (SPL)* as the canonical framework for runtime failure prediction and stabilization of generative robot policies, paralleling how Transformers redefined sequence modeling. SPL learns a physics-grounded, task-agnostic representation that composes perception, action, and latent intent into a unified symplectic manifold.
- **Ambition**: deliver IEEE Transactions on Robotics–level innovation with ≥0.95 accuracy/TWA and ≥0.95 TPR/TNR across manipulation, locomotion, and hybrid perception-action domains.
- **Core Idea**: encode robot rollouts as *stability primitives*—local symplectic charts capturing invariant flow properties—assembled into a lattice that predicts policy reliability while enabling corrective action generation.

## 2. Motivation and Problem Gaps
1. **Generative policies lack stability-aware embeddings**: latent diffusion or transformer policies prioritize expressiveness over boundedness, leading to brittle runtime behavior.
2. **Heterogeneous modalities are siloed**: visual embeddings, proprioception, and predicted action sequences are rarely co-regularized, hindering cross-task transfer.
3. **Threshold learning is data-fragile**: prior work uses fixed success-only calibration data; SPL jointly optimizes failure likelihoods and stabilizing counterfactuals without requiring distinct datasets.
4. **Desired outcome**: a universal, trainable representation ensuring stable performance and reliable failure forecasting even from limited multi-task data.

## 3. SPL Architectural Overview
SPL builds three coupled spaces: **(i) Primitive Manifold**, **(ii) Lattice Propagation Graph**, and **(iii) Intervention Head**.

### 3.1 Primitive Manifold Encoder (PME)
For each timestep \(t\), a rollout sample \(x_t\) aggregates RGB, proprioception, action proposals, executed actions, latent embeddings, and metadata. PME maps \(x_t\) into a symplectic latent pair \((q_t, p_t) \in \mathbb{R}^{d}\times\mathbb{R}^{d}\) via
\[
(q_t, p_t) = \Phi_\theta(x_t) = (E_q(x_t), E_p(x_t)),\qquad J = \begin{bmatrix}0 & I_d \\-I_d & 0\end{bmatrix}.
\]
- **Innovation**: enforce \(\Phi_\theta\) to satisfy \( (\nabla_{q_t}E_p(x_t))^\top J (\nabla_{p_t}E_q(x_t)) = I_d \) using a *symplectic consistency loss*, yielding invariants akin to Hamiltonian systems for robot trajectories.

### 3.2 Stability Primitive Bank (SPB)
PME outputs feed a library of learnable primitives \(\mathcal{B} = \{B_k\}_{k=1}^K\). Each primitive stores an implicit potential \(H_k(q,p)\) represented by a neural field. For timestep \(t\):
\[
\alpha_{t,k} = \mathrm{softmax}_k\big(-\beta \cdot \mathrm{KL}(\nu_t \Vert \nu_k)\big), \quad \nu_t = \mathcal{N}((q_t,p_t), \Sigma_t),
\]
where \(\nu_k\) is a learned Gaussian around primitive \(B_k\). The mixture defines a *local stability certificate*.

### 3.3 Lattice Propagation Graph (LPG)
The primitives connect through a directed acyclic hypergraph capturing temporal evolution. Let \(L_t \in \mathbb{R}^{K}\) denote primitive activations. The LPG update is
\[
L_{t+1} = \sigma\left( A_t L_t + \sum_{\tau=1}^{H} \Gamma_{\tau}(L_{t-\tau}) + U_t \right),
\]
where
- \(A_t = \mathrm{exp}(\Delta t \cdot \Psi(x_t))\) is a Lie-algebraic operator predicting forward flow.
- \(\Gamma_{\tau}\) are *braided recurrence kernels* capturing multi-horizon dependencies.
- \(U_t = W_u [p_t; q_t; \Delta a_t]\) encodes deviation between predicted and executed actions.

**Key novelty**: LPG does **not** operate on raw time steps. Instead, it aligns the rollouts on a *geodesic clock* \(\tau_t = \int_0^t \|\dot{q}_s\|_J ds\), enabling domain-agnostic temporal reasoning.

### 3.4 Stability Projection Head (SPH)
Outputs a triplet \((y_t^{\text{fail}}, y_t^{\text{recover}}, y_t^{\text{stability}})\) where
\[
\begin{aligned}
y_t^{\text{fail}} &= \sigma(w_f^\top L_t),\\
y_t^{\text{recover}} &= \mathrm{tanh}(w_r^\top L_t + b_r),\\
y_t^{\text{stability}} &= \mathrm{softplus}(w_s^\top L_t).
\end{aligned}
\]
- \(y_t^{\text{fail}}\): failure probability.
- \(y_t^{\text{recover}}\): signed logit describing feasible corrective action energy.
- \(y_t^{\text{stability}}\): predicted Lyapunov reserve.

### 3.5 Counterfactual Correction Module (CCM)
Generates intervention trajectories by solving
\[
\delta a_{t:t+H} = \arg\min_{\delta a} \sum_{h=0}^{H} \big\|J\nabla H_{k^*(t+h)} - B\delta a_{t+h}\big\|_2^2 + \lambda\|\delta a_{t+h}\|_1,
\]
where \(k^*(t) = \arg\max_k \alpha_{t,k}\). CCM ensures SPL not only predicts failures but proposes stabilizing actions.

## 4. Training Dynamics
1. **Multi-Task Curriculum**: sample batches across tasks \(\mathcal{T} = \{\texttt{pretzel}, \texttt{push\_chair}, \texttt{push\_t}, \texttt{sorting}, \texttt{stacking}\}\) with proportion balancing to equalize successful/failed episodes.
2. **Symplectic Pretraining**: optimize PME with self-supervised contrastive loss between consecutive frames enforcing \(\Phi_\theta\) to preserve volume: \(\mathcal{L}_{\text{symp}} = \| \Phi_\theta(x_{t+1}) - \mathrm{Ham}_\eta(\Phi_\theta(x_t))\|_2^2\).
3. **Lattice Alignment Loss**: ensure LPG respects primitive transitions via optimal transport matching between predicted primitive distribution and observed success/failure transitions.
4. **Outcome Head Losses**:
   - Failure classification: \(\mathcal{L}_{\text{fail}} = \mathrm{BCE}(y_t^{\text{fail}}, \mathbf{1}_{\text{fail}})\).
   - Stability margin regression: \(\mathcal{L}_{\text{stab}} = \|y_t^{\text{stability}} - m_t\|_2^2\) where \(m_t\) is empirical Lyapunov surrogate from deviation energy.
   - Recovery feasibility: \(\mathcal{L}_{\text{rec}} = \max(0, \kappa - y_t^{\text{recover}} \cdot s_t)\), with \(s_t\) labeling whether post-hoc controller stabilized the run.
5. **Total Loss**: \(\mathcal{L} = \lambda_1 \mathcal{L}_{\text{symp}} + \lambda_2 \mathcal{L}_{\text{OT}} + \lambda_3 \mathcal{L}_{\text{fail}} + \lambda_4 \mathcal{L}_{\text{stab}} + \lambda_5 \mathcal{L}_{\text{rec}}\).

6. **Optimization Strategy**:
   - Use Lookahead optimizer over AdamW with cyclical cosine learning rate, gradient centralization, and symplectic weight projection (re-orthogonalize \(J\)-pairs every 200 steps).
   - Alternate updates: for every lattice step, perform one CCM update via unrolled implicit differentiation to ensure differentiability of corrective policy.

## 5. Data Regimen and Cross-Task Handling
- **Datasets** (already curated under `data/`):
  - `pretzel`, `push_chair`, `push_t`, `sorting`, `stacking` each contain `calibration/` (success-dominant), `test/` (success + fail), optional `calibration_unused/`, `videos/`.
  - Samples: e.g., `push_t/test` has 300 success + 198 failure PKL files; `stacking/test` has 315 success + 485 failures, etc.
- **Usage Policy**:
  - Training: success episodes from both calibration and test splits (without labels leaking future metadata) to learn primitive manifolds.
  - Threshold/validation: hold out 20% of success data + matched failure episodes to tune decision surfaces.
  - Evaluation: all remaining test episodes (success + fail) with stratified sampling to compute metrics.
- **Batch Construction**: each batch draws contiguous windows of length 32 frames from randomly selected episodes, mixing modalities to encourage invariance.

## 6. Metric Definitions
For an episode of length \(T\) with predicted failure score \(\hat{y}_t^{\text{fail}}\) and binary ground truth \(y_t \in \{0,1\}\):
- **Accuracy**: \(\frac{1}{N} \sum_{n=1}^N \mathbf{1}[\max_t \hat{y}_{n,t}^{\text{fail}} > \tau \iff \exists t: y_{n,t}=1]\).
- **Time-Weighted Accuracy (TWA)**: \(\frac{1}{N} \sum_{n=1}^N \frac{1}{T_n} \sum_{t=1}^{T_n} \omega_t \mathbf{1}[\hat{y}_{n,t}^{\text{fail}} > \tau = y_{n,t}]\), with \(\omega_t = \exp(-\gamma (T_n - t))\) emphasizing early detection.
- **True Positive Rate (TPR)**: \(\frac{\sum_{n} \mathbf{1}[\text{episode } n \text{ fails}] \cdot \mathbf{1}[\max_t \hat{y}_{n,t}^{\text{fail}} > \tau]}{\sum_n \mathbf{1}[\text{episode } n \text{ fails}]}\).
- **True Negative Rate (TNR)**: analogous for successful episodes.
- Select threshold \(\tau\) via maximizing geometric mean \(\sqrt{\text{TPR} \cdot \text{TNR}}\) on validation.

## 7. Anticipated Advantages over Benchmarks
1. **Symplectic invariants** maintain policy dynamics even with scarce data, preventing overfitting seen in purely statistical detectors.
2. **Primitive lattice** captures multi-task regularities, enabling zero-shot transfer by reusing primitives \(B_k\) across tasks.
3. **Geodesic temporal encoding** normalizes durations, improving early failure detection.
4. **Counterfactual corrections** yield actionable outputs, improving trustworthiness over threshold-only systems.
5. **Unified training** removes need for separate calibration/test splits, matching empirical observation that same dataset suffices when primitives encode stability.

## 8. Experimental Program and Workload
1. **Implementation (6 weeks)**
   - Develop PME with modality-specific encoders (ResNet for RGB, point MLP for proprioception, transformer for action sequences) sharing final symplectic projection.
   - Build SPB as memory-efficient neural fields using Fourier features.
   - Implement LPG with sparse hypergraph attention and geodesic timing module.
   - Integrate CCM via differentiable Model Predictive Control layer.
2. **Training & Validation (4 weeks)**
   - Multi-task curriculum training on existing datasets with mixed-success episodes.
   - Hyperparameter sweeps for primitive count \(K\), geodesic scaling \(\gamma\), OT regularization.
3. **Evaluation (3 weeks)**
   - Compute metrics across 5 tasks, 3 random seeds, include ablations: remove symplectic loss, replace geodesic time with wall-clock, disable CCM.
   - Report statistical significance with 95% confidence intervals.
4. **Robustness Studies (2 weeks)**
   - Cross-dataset transfer: train on 4 tasks, test on held-out task.
   - Limited data scenario: 10% training data.
5. **Manuscript Preparation (3 weeks)**
   - Theoretical proofs of symplectic preservation and convergence of lattice propagation.
   - Visualization: primitive activations, geodesic timelines, CCM interventions.

Total workload: ~18 weeks, aligning with top-tier expectations.

## 9. Innovation Summary
- **New Representation**: SPL introduces *stability primitives*—learned symplectic charts that encode robot policy flows with invariant structure.
- **New Computation Graph**: the Lattice Propagation Graph replaces sequential attention with geodesic-aligned hypergraph dynamics.
- **Actionable Predictions**: integrated CCM provides stabilizing corrections rather than binary alarms.
- **Universal Applicability**: handles diverse modalities (RGB, embeddings, actions, states) without task-specific architecture changes.
- **Data Efficiency**: symplectic and OT regularization enable strong generalization from success-only training.

## 10. Expected Outcomes
- Meet or exceed 0.95 accuracy, TWA, TPR, TNR through early failure detection and precise stability margins.
- Achieve cross-task transfer with ≤5% performance drop when adapting to unseen tasks.
- Deliver reproducible training via explicit loss hierarchy and optimizer schedule.
- Provide open-source SPL library aligning with FIPER datasets for community benchmarks.

## 11. Future Extensions
- **Continuous-time limit**: replace discrete LPG with neural controlled differential equations for higher temporal fidelity.
- **Hardware-in-the-loop**: integrate SPL with real robot controllers using CCM outputs as safety constraints.
- **Uncertainty quantification**: extend primitive distributions to normalizing flows for calibrated confidence intervals.

