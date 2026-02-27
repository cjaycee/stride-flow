# StrideFlow: Running Form Analyzer

StrideFlow is a computer vision-based tool designed to analyze running biomechanics from video footage.

## Project Goals

The primary objective of this project is to track key running metrics to help athletes improve their form and reduce injury risk:

*   **Cadence Tracking**: Calculate steps per minute (SPM).
*   **Knee Angle Analysis**: Measure maximum and minimum knee flexion/extension during the gait cycle.
*   **Vertical Oscillation**: Track the vertical movement of the center of mass or hips.

## Project Structure

*   `src/`: Source code for video processing and analysis.
*   `data/videos/`: Raw video files for analysis.
*   `output/`: Analyzed videos and data logs.

## Setup

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   ```
2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
