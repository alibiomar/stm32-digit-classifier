# 🧠 STM32 Digit Classifier

A real-time handwritten digit recognition system that runs a neural network directly on an STM32 microcontroller. Draw digits on a desktop GUI, and watch as the STM32 processes them using on-chip AI inference powered by X-CUBE-AI.

![STM32](https://img.shields.io/badge/STM32-03234B?style=flat&logo=stmicroelectronics&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20Lite-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- 🎨 **Interactive Drawing Canvas** - User-friendly Tkinter GUI for drawing digits (0-9)
- 🚀 **Edge AI Computing** - Neural network inference runs entirely on the STM32 microcontroller
- ⚡ **Real-time Classification** - Fast prediction with minimal latency
- 🔌 **Serial Communication** - Seamless UART connection between PC and STM32
- 📦 **Standalone Executable** - No Python installation required for end users
- 🎯 **High Accuracy** - Trained on EMNIST dataset for robust digit recognition

## 🛠️ Hardware Requirements

- **STM32F411VET6** microcontroller board (or compatible STM32F4 series)
- USB-to-Serial adapter or STM32's built-in USB
- Windows PC (for GUI application)

## 📋 Software Requirements

### For Running the Application:
- Windows 10/11
- USB serial drivers (usually auto-installed)

### For Development:
- **Python 3.8+** with packages:
  - `tkinter`
  - `Pillow`
  - `numpy`
  - `pyserial`
  - `PyInstaller` (for building executable)
- **STM32CubeIDE** or compatible ARM toolchain
- **STM32CubeMX** with X-CUBE-AI expansion pack

## 🚀 Quick Start

### Using Pre-built Executable

1. Download the latest release from the [Releases](../../releases) page
2. Connect your STM32 device via USB
3. Run `STM32_Digit_Classifier.exe`
4. Configure the COM port and baud rate (default: COM9, 115200)
5. Click "Connect to Device"
6. Draw a digit and click "Predict Digit"

### Running from Source

```bash
# Clone the repository
git clone https://github.com/alibiomar/stm32-digit-classifier.git
cd stm32-digit-classifier

# Install dependencies
pip install pillow numpy pyserial

# Run the application
python main.py
```

## 📁 Project Structure

```
tinyML/
├── main.py                          # Python GUI application
├── emnist_digits_int8.tflite       # Quantized TFLite model
├── STM32_Digit_Classifier.spec     # PyInstaller configuration
├── tinyML.ipynb                     # Jupyter notebook (training/analysis)
├── tinyML/                          # STM32 firmware project
│   ├── Core/
│   │   ├── Src/
│   │   │   └── main.c              # Main firmware code
│   │   └── Inc/
│   │       └── main.h              # Header files
│   ├── X-CUBE-AI/                  # AI middleware
│   │   └── App/
│   │       ├── network.c           # Generated neural network
│   │       ├── network_data.c      # Model weights
│   │       └── network.h
│   ├── Drivers/                     # HAL and CMSIS drivers
│   └── tinyML.ioc                  # STM32CubeMX configuration
└── README.md
```

## 🔬 How It Works

### 1. Model Training & Conversion
- Neural network trained on the EMNIST dataset
- Converted to TensorFlow Lite format
- Quantized to INT8 for optimal performance on MCU

### 2. STM32 Integration
- TFLite model converted to C code using X-CUBE-AI
- Embedded in STM32 firmware
- Inference runs on Cortex-M4 core

### 3. Communication Protocol
- GUI captures 28×28 pixel digit image
- Image data sent via UART (115200 baud)
- STM32 processes and returns predicted digit
- GUI displays result in real-time

### 4. Inference Pipeline
```
Draw Digit → Preprocess (28×28) → Send via UART → STM32 Inference → Return Prediction → Display Result
```

## 🔧 Building from Source

### Building the Executable

```bash
# Install PyInstaller
pip install pyinstaller

# Build standalone executable
pyinstaller --onefile --windowed --name "STM32_Digit_Classifier" --hidden-import PIL._tkinter_finder main.py

# Executable will be in dist/ folder
```

### Flashing STM32 Firmware

1. Open `tinyML/tinyML.ioc` in STM32CubeMX
2. Generate code
3. Open in STM32CubeIDE
4. Build the project
5. Flash to STM32 board using ST-Link

## 🎨 GUI Preview

The application features a modern, responsive interface with:
- **Connection Screen** - Configure and establish serial connection
- **Drawing Canvas** - 320×320 pixel drawing area with smooth brush
- **Result Display** - Clear visualization of predicted digit

## ⚙️ Configuration

### Serial Settings
- **Port**: Adjust to your STM32's COM port (check Device Manager)
- **Baud Rate**: 115200 (default, can be changed in firmware)
- **Timeout**: 5 seconds for data transmission

### Model Parameters
- **Input**: 28×28 grayscale image (784 pixels)
- **Output**: Digit 0-9
- **Quantization**: INT8
- **Model Size**: ~10KB

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **STMicroelectronics** for X-CUBE-AI middleware
- **TensorFlow** team for TensorFlow Lite
- **EMNIST** dataset creators

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project is designed for educational purposes to demonstrate edge AI and embedded machine learning concepts.
