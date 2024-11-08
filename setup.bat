@echo off
echo Setting up environment and installing dependencies...

REM Create a virtual environment if it doesn't already exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
call venv\Scripts\activate

REM Upgrade pip to the latest version
echo Upgrading pip...
pip install --upgrade pip

REM Check for CUDA version using nvcc
echo Checking for CUDA support using nvcc...
nvcc --version > cuda_check.txt 2>nul

REM Check if nvcc command was successful
if %errorlevel% NEQ 0 (
    echo CUDA not found. Setting CUDA version to CPU.
    set "cuda_version=CPU"
) else (
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
)

REM Display detected CUDA information
echo CUDA Version Detected: %cuda_version%

REM Create or overwrite the requirements.txt file
echo Creating requirements.txt...
(
    echo torch
    echo torchvision
    echo datasets
    echo transformers
    echo tokenizers
    echo matplotlib
    echo scipy
    echo tensorboard
    echo psutil
    echo opencv-python
) > requirements.txt

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
if exist cuda_check.txt del cuda_check.txt

REM Install other necessary libraries from requirements.txt
echo Installing other dependencies from requirements.txt...
pip install -r requirements.txt

REM Verify installation of transformers
echo Verifying installation of transformers...
pip show transformers >nul 2>&1
if %errorlevel% NEQ 0 (
    echo transformers not found. Installing transformers...
    pip install transformers
    if %errorlevel% NEQ 0 (
        echo Failed to install transformers. Please check the error messages above.
        pause
        exit /b 1
    )
) else (
    echo transformers is already installed.
)


REM Download the MNIST dataset
echo Downloading MNIST dataset...
python -c "from torchvision import datasets; datasets.MNIST('./data', train=True, download=True); datasets.MNIST('./data', train=False, download=True)"

REM Download the CIFAR-10 dataset
echo Downloading CIFAR-10 dataset...
python -c "from torchvision import datasets; datasets.CIFAR10('./data', train=True, download=True); datasets.CIFAR10('./data', train=False, download=True)"

REM Download the IMDB dataset using Hugging Face's datasets library
echo Downloading IMDB dataset...
python -c "from datasets import load_dataset; load_dataset('imdb', split='train').to_pandas().to_csv('./data/imdb_train.csv'); load_dataset('imdb', split='test').to_pandas().to_csv('./data/imdb_test.csv')"

REM Download the AG_NEWS dataset using Hugging Face's datasets library
echo Downloading AG_NEWS dataset...
python -c "from datasets import load_dataset; load_dataset('ag_news', split='train').to_pandas().to_csv('./data/ag_news_train.csv'); load_dataset('ag_news', split='test').to_pandas().to_csv('./data/ag_news_test.csv')"

REM Download the Pascal VOC dataset
echo Downloading Pascal VOC dataset...
python -c "from torchvision import datasets; datasets.VOCSegmentation(root='./data', year='2012', image_set='train', download=True); datasets.VOCSegmentation(root='./data', year='2012', image_set='val', download=True)"


REM Inform the user that setup is complete
echo Setup complete. You can now run the main.py script.
pause
