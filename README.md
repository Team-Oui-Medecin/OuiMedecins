## Prerequisites
- Python 3.11 (required)
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

```

### 4. Select Python Interpreter in VSCode
- Press `Cmd/Ctrl + Shift + P`
- Type "Python: Select Interpreter"
- Choose `./venv/bin/python` (Python 3.11.x)

## Resources
- [Inspect AI Documentation](https://ukgovernmentbeis.github.io/inspect_ai/)
- [Team Wiki](link-to-your-wiki)