# Polymander Ground Reaction Force Estimation

Polymander is a salamander-inspired amphibious robot developed at EPFL's BioRob lab, able to both walk and swim.

![Demo](assets/videos/walking-polymander.gif)

*Demonstration of the Polymander robot walking over force plates while motor currents and ground reaction forces are recorded.*

▶️ **Full demonstration:** [assets/videos/walking-polymander.mp4](assets/videos/walking-polymander.mp4)

## Motivation
In case of real world scenario such as disaster recovery, legged robots can navigate in challenging and complex environments where wheeled robots may struggle. To achieve stable locomotion, robots rely on sensory information like **ground reaction force (GRF)**, commonly measured with dedicated force/torque sensors mounted on their feet. However, these additional sensors increase hardware complexity, weight, and cost, while potentially reducing reliability and durability of the robot due to repeated impact forces during locomotion.

Can GRFs be estimated directly from **motor current measurements** instead, without a dedicated force sensor?

Using synchronized motor current and force-plate recordings, this pipeline filters and aligns the two signals, then trains a regression model to predict vertical GRF (Fz) from current alone, removing the need for a dedicated force sensor at inference time.

**Software:** Python · NumPy · pandas · SciPy · scikit-learn · Signal Processing · Multiple Linear Regression

**Hardware:** Raspberry Pi Zero W · Dynamixel Actuators · Kistler Force Plates 

## Experimental Setup

Experiments were conducted on the **Polymander** quadruped robot, equipped with **16 Dynamixel actuators** and controlled by a **Raspberry Pi Zero W**. Two independent systems recorded data simultaneously during each trial:

- **Motor currents:** logged on-board via the Raspberry Pi Zero W as the robot walked.
- **Ground reaction forces:** measured by a **Kistler force plate**, acquired through the lab's DAQ system and logged on a lab laptop, serving as ground truth.

![Robot](assets/images/polymander.png)

*Polymander quadruped robot.*

![Force plate setup](assets/images/force-plate-setup.png)

*Experimental setup for recording ground reaction forces.*

Motor control and data logging rely on a C++ controller running locally on the robot's Raspberry Pi Zero W, originally developed by **Laura Paez**. I extended the controller to implement the stepping sequence, and the initial-posture logic that starts the test limb pre-lifted off the ground. These extensions were used to generate the motor feedback logs used in this project. The controller itself is not included in this repo (lab-owned).

The two data streams (motor currents and force plate measurement) were recorded at different sampling rates and synchronized during post-processing.

## Pipeline

The complete workflow consisted of four stages:
1. Record raw signals
2. Filtering and synchronization
3. Multiple linear regression
4. Force estimation

![Pipeline](assets/images/pipeline.png)

## Results

**R² ≈ 0.8**

Combining hip and calf motor currents produced the most accurate force estimates. The calf actuator (pitch movement -> related to the Fz force) is the stronger predictor; combining both gives the best result.

The predicted force closely follows the measured force plate signal throughout the gait cycle.

![Prediction](assets/images/mlr-pred.png)

|First view | Second view | Third view|
|---|---|---|
![mlr-hyperplane-1](assets/images/mlr-hyperplane-1.png) | ![mlr-hyperplane-2](assets/images/mlr-hyperplane-2.png)| ![mlr-hyperplane-3](assets/images/mlr-hyperplane-3.png)

*Three views of the fitted multiple linear regression plane relating hip and calf motor currents to the measured vertical ground reaction force.*

**Limitations**
- Linear regression cannot capture all nonlinear actuator dynamics, which shows up as residual error (RMSE ≈ 1N on a 0-4N range).
- Experiments were also limited to 45 trials at 3 amplitudes and 3 frequencies for a single stepping limb (front left) rather than full walking gait. Walking only produced one step on the force plate per trial, which wasn't enough data to train on.

## Repository Structure

```text
polymander-grf-estimation/
├── assets/
│   ├── images/
│   └── videos/
├── data/                    # not included - lab-owned experimental data
├── models/                  # trained regression model (.pkl)
├── notebooks/
├── polymander/              # loaders, signal processing, visualization
├── results/                 # not included - output figures
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Getting Started
If you've ever inherited a project with no setup instructions and burned an afternoon on a version mismatch, this section is for you.

### Installation

Clone the repo and setup a conda environment:

```bash
git clone https://github.com/inalbon/polymander-grf-estimation.git
cd polymander-grf-estimation
conda create -n polymander python=3.10
conda activate polymander
```

Install dependencies and the local package:

```bash
pip install -r requirements.txt
pip install -e .
```

### Running
```bash
  python scripts/train_force_estimator.py
```

> Note: `scripts/` and `notebooks/` expect the original experimental data, which isn't included in this repo (lab-owned). The trained models in `models/` are provided directly so you can inspect results without needing to rerun the pipeline. Feel free to reach out if you're curious about the data.

## Authors & Acknowledgements

**Malika In-Albon**

Semester project (10 ECTS, Fall 2022) at the **Biorobotics Laboratory (BioRob)** supervised by **Astha Gupta**, under **Prof. Auke Ijspeert**.
