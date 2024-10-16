@echo off
echo Setting up environment and installing dependencies...

REM Create a virtual environment if it doesn't already exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Check for CUDA version using nvcc
echo Checking for CUDA support using nvcc...
nvcc --version > cuda_check.txt

REM Read the CUDA version from nvcc output
set "cuda_version=CPU"
for /f "tokens=5 delims= " %%A in ('findstr /r /c:"release" cuda_check.txt') do (
    set "cuda_version=%%A"
)

REM Extract only the major and minor version (e.g., 12.4) and remove any trailing commas or extra characters
for /f "tokens=1,2 delims=." %%A in ("%cuda_version:,=%") do (
    set "cuda_version_major=%%A"
    set "cuda_version_minor=%%B"
)
set "cuda_version=%cuda_version_major%.%cuda_version_minor%"

REM Display detected CUDA information
echo CUDA Version Detected: %cuda_version%

REM Create a requirements.txt file if it does not exist
echo torch > requirements.txt
echo torchvision >> requirements.txt
echo matplotlib >> requirements.txt

REM Install the appropriate version of PyTorch based on CUDA availability
if /i "%cuda_version%" == "CPU" (
    echo No CUDA device detected or CUDA not available. Installing the CPU version of PyTorch...
    pip install torch torchvision
) else (
    echo CUDA detected. Installing the CUDA version of PyTorch for version %cuda_version%...
    if "%cuda_version%" == "12.4" (
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    ) else (
        echo Unrecognized or unsupported CUDA version detected. Proceeding with default CUDA installation...
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    )
)

REM Cleanup the CUDA check file
del cuda_check.txt

REM Install other necessary libraries
pip install -r requirements.txt

REM Download the MNIST dataset
echo Downloading MNIST dataset...
python -c "from torchvision import datasets; datasets.MNIST('./data', train=True, download=True); datasets.MNIST('./data', train=False, download=True)"

echo Setup complete. You can now run the main.py script.
pause
