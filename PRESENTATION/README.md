# Beamer Presentation: Deep Reinforcement Learning for EV Routing Optimization

## Overview

This folder contains the LaTeX Beamer presentation for the **Energy-Minimizing Electric Vehicle Routing Problem (EM-EVRP)** project using Deep Reinforcement Learning.

## Contents

- `main.tex` - Main Beamer presentation source file (33 slides)
- Image files (PNG format) - Figures used in the presentation

## Slide Structure (33 slides total)

1. **Title Slide**
2. **Outline**

### Introduction (Slides 3-5)
3. Context: The Rise of Electric Vehicles
4. Challenges in EV Logistics
5. Motivation: Why This Project?

### Problem Statement (Slides 6-8)
6. Electric Vehicle Routing Problem (EVRP)
7. Energy-Minimizing EVRP (EM-EVRP)
8. Why Minimize Energy Instead of Distance?

### Mathematical Formulation (Slides 9-11)
9. MILP Formulation: Sets and Parameters
10. Physics-Based Energy Consumption Model
11. MILP Formulation: Objective and Constraints

### Baseline Optimization Methods (Slides 12-14)
12. Classical Optimization Approaches
13. Metaheuristics: Local Search Operators
14. Ant Colony Optimization

### Deep Reinforcement Learning Approach (Slides 15-21)
15. Why Deep Reinforcement Learning?
16. MDP Formulation for EVRP
17. Attention-Based Architecture: Overview
18. Graph Attention Encoder
19. Context-Aware Decoder
20. REINFORCE Training Algorithm
21. EV-Specific Model Features

### System Architecture (Slides 22-27)
22. End-to-End System Pipeline
23. Data Generation Pipeline
24. Real-World Deployment: OSRM Integration
25. Interactive Dashboard: Main Interface
26. Interactive Dashboard: Route Optimization
27. Interactive Dashboard: Analytics

### Results and Evaluation (Slides 28-30)
28. Training Configuration
29. Training Results
30. Performance Comparison
31. Key Findings

### Conclusion (Slides 31-33)
32. Summary of Contributions
33. Limitations and Future Work
34. Conclusion
35. Thank You / Questions

### Backup Slides (Appendix)
- Node Indexing Convention
- Masking Logic Details
- References

## Compilation Instructions

### Requirements

- LaTeX distribution (TeX Live, MiKTeX, or MacTeX)
- Required packages:
  - beamer
  - tikz, pgfplots
  - amsmath, amssymb
  - booktabs
  - fontawesome5
  - hyperref

### Compile with pdflatex

```bash
# Navigate to PRESENTATION folder
cd PRESENTATION

# Compile (run twice for references)
pdflatex main.tex
pdflatex main.tex
```

### Compile with latexmk (recommended)

```bash
latexmk -pdf main.tex
```

### Using VS Code with LaTeX Workshop

1. Open `main.tex` in VS Code
2. Install LaTeX Workshop extension
3. Press `Ctrl+Alt+B` to build
4. The PDF will be generated automatically

## Design Features

- **Theme**: Madrid with custom whale color scheme
- **Aspect Ratio**: 16:9 (widescreen)
- **Custom Colors**: 
  - Primary Blue (RGB: 0, 84, 147)
  - Secondary Blue (RGB: 0, 119, 182)
  - Accent Green (RGB: 0, 150, 136)
- **Professional Footer**: Author, title, slide number
- **TikZ Diagrams**: Custom illustrations throughout
- **Consistent Typography**: Clean, readable fonts

## Image Files Used

| File | Description |
|------|-------------|
| `main_page.PNG` | Dashboard main interface |
| `inference.PNG` | Inference process |
| `result.PNG` | Optimized routes |
| `analytics.PNG` | Analytics view |
| `detailed.PNG` | Detailed analysis |
| `drl_model_architecture.png` | Model architecture diagram |
| `system_architecture.png` | System pipeline |
| `synthetic_training_data.png` | Training data visualization |
| `training_reward.png` | Training curves |

## Authors

- **Salem Fradi**
- **Dhia Elhak Ezzeddini**

## Supervisor

- **Dr. Ferdaoues Chaabane**

## Institution

- Higher School of Communication of Tunis (Sup'Com)
- Carthage University

## Host Company

- Sagemcom

## Academic Year

2025-2026
