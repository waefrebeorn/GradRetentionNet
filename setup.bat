@echo off
echo Setting up environment and installing dependencies...

REM Create a virtual environment if it doesn't already exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Create a requirements.txt file if it does not exist
echo torch > requirements.txt
echo torchvision >> requirements.txt
echo matplotlib >> requirements.txt

REM Install the necessary libraries
pip install -r requirements.txt

REM Download the MNIST dataset
echo Downloading MNIST dataset...
python -c "from torchvision import datasets; datasets.MNIST('./data', train=True, download=True); datasets.MNIST('./data', train=False, download=True)"

echo Setup complete. You can now run the main.py script.
pause
