# Railway Platform Safety Control System (FYP Project)

A real-time computer vision system designed to enhance safety on railway platforms by detecting dangerous situations and alerting operators. This system combines **zone segmentation** and **person detection** to monitor platform areas and identify safety risks.

## Features

- **Zone Segmentation**: Automatically detects and classifies platform areas using YOLOv11:
  - 🔴 **Danger Zones** (red danger areas near tracks)
  - 🟡 **Yellow Line Zones** (warning areas)
  - 🟢 **Safe Zones** (safe passenger areas)

- **Person Detection**: Real-time detection and tracking of people using RF-DETR:
  - Identifies individuals in danger zones
  - Counts people in warning areas
  - Generates risk metrics

- **Operator Interface**:
  - Desktop GUI for real-time monitoring
  - Live visualization of input and labeled output
  - Operational metrics and incident logging
  - Zoom and pan capabilities for detailed inspection
  - Emergency alerts with sound warnings

- **Safety Alerts**:
  - Visual and sound alerts for danger zone intrusion  
  - Warning alerts for yellow line violations

- **Comprehensive Operational Metrics**:
  - View detection results
  - Assess system performance (accuracy, response speed)

## System Requirements
- **Python**: 3.11+
- **OS**: Windows, Linux, or macOS

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TeoJJss/platform-safety.git
cd platform-safety
```

### 2. Set Up Virtual Environment

```bash
# Create virtual environment
python -m venv venv3.11
```

**Activate virtual environment:**

**On Windows:**
```bash
venv3.11\Scripts\activate
```

**On Linux/macOS:**
```bash
source venv3.11/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: This may take several minutes as it includes PyTorch, Ultralytics, and other ML libraries.  

### 4. View the AI models    
The model development (annotation, training etc) was done at <b>Roboflow</b>.  
Refer to `dataset.txt` for the dataset and models.  


## Quick Start

### Running the Application

**Activate virtual environment (if not already active):**

**Windows:**
```bash
venv3.11\Scripts\activate
```

**Linux/macOS:**
```bash
source venv3.11/bin/activate
```

**Run the application:**
```bash
python platform_safety_gui.py
```

The application window will open with the Railway Platform Safety Control Center interface.

### Using the Application

1. **Load Input**:
   - Click "Select Image / Video" to choose a file
   - Supported formats: `.jpg`, `.jpeg`, `.png`, `.mp4`

2. **Configure Settings**:
   - Check "Show Zone Masks" to visualize segmentation overlays
   - Customize other operational parameters as needed

3. **Process**:
   - The application automatically starts processing after selection
   - Monitor real-time metrics on the right panel
   - View incident logs for alerts and warnings

4. **Zoom & Pan**:
   - Use mouse wheel to zoom in/out on input or output panels
   - Hover over panels to inspect details

5. **Stop**:
   - Click "Stop" button to halt ongoing processing

## Credits
- <a href="https://github.com/TeoJJss">Teo Jun Jia</a>
