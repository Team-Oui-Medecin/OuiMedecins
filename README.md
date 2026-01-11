# View Our Latest Results
- [Explore them here]([url](https://team-oui-medecin.github.io/OuiMedecins/#/logs/ 
))

## Prerequisites
- Python 3.11 or 3.12 (required)
- Git

## Setup Instructions

### 1. Verify Python Version
```bash
python --version  # Should show Python 3.11.x
```

If you don't have Python 3.11, install it:
- **macOS**: `brew install python@3.11`
- **Ubuntu/Debian**: `sudo apt install python3.11 python3.11-venv`
- **Windows**: Download from [python.org](https://www.python.org/downloads/)

### 2. VSCode Setup
```bash
# Clone the repository
git clone OuiMedecines

```

### 3. Environment Setup
```bash
# Create virtual environment with Python 3.11
python3.11 -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate    # Windows

# Install the requirements (includes inspect-ai)
pip install -r requirements.txt

```

### 4. Select Python Interpreter in VSCode
- Press `Cmd/Ctrl + Shift + P`
- Type "Python: Select Interpreter"
- Choose `./venv/bin/python` (Python 3.11.x)

## How to work with Inspect

### Run the benchmark
```bash
python eval.py
```

### View the benchmark run (log)
You can view it via the VSCode extension "Inspect AI"
Alternative way:
```bash
# generate a html page
inspect view bundle   --log-dir ./logs   --output-dir ./logs_www --overwrite
# start a local webserver
python -m http.server 8080
# go to http://localhost:8080 in your browser then choose "logs_www". you should see benchmark runs now.
```

### Run the visualization
```bash
python data_for_visualization.py
```

### View visualization
```bash
# start a local webserver
python -m http.server 8080
# go to http://localhost:8080/visualization.html in your browser and you should see the data visualization.
```

## Resources
- [Inspect AI Documentation](https://ukgovernmentbeis.github.io/inspect_ai/)
- [Team Wiki](link-to-your-wiki)

