# Bulk Image Processing with CUDA

A beginner CUDA C++ project that converts every supported image in an input folder to grayscale. The CPU loads and saves the files, while a CUDA kernel processes the pixels in parallel on the GPU.

## Features

- Batch-processes images in a folder.
- Reads PNG, JPEG, and BMP files through `stb_image`.
- Converts each image to RGB before GPU processing.
- Uses one CUDA thread per pixel and weighted luminance:

  ```text
  gray = 0.299 × red + 0.587 × green + 0.114 × blue
  ```

- Saves one-channel grayscale PNG files with `_gray` appended to each filename.
- Reports CUDA runtime errors with a readable message.

## Requirements

- An NVIDIA GPU and a working CUDA Toolkit with `nvcc`.
- A C++17-capable compiler (used for `std::filesystem`).
- `stb_image.h` and `stb_image_write.h` in the same directory as `grayscale.cu`.

The code can be developed and run in Google Colab with a GPU runtime. GPU availability varies by session. The notebook runtime is temporary, so keep a copy of the `.cu` source and header files somewhere persistent.

## Build

In Colab, create `grayscale.cu` from the source cell, make sure both stb headers are available in the working directory, then compile:

```bash
!nvcc -std=c++17 -O2 grayscale.cu -o grayscale
```

For a local CUDA installation, run the equivalent command in a terminal:

```bash
nvcc -std=c++17 -O2 grayscale.cu -o grayscale
```

## Run

The program takes an input folder and an output folder:

```bash
./grayscale input output
```

For example, if `input/` contains `photo.png` and `scene.jpg`, the program writes:

```text
output/photo_gray.png
output/scene_gray.png
```

The output folder is created automatically if it does not exist. Non-image files and files with extensions other than `.png`, `.jpg`, `.jpeg`, and `.bmp` are skipped. Extension matching is currently case-sensitive.

In Colab, invoke it from a notebook cell with:

```bash
!mkdir -p input
!./grayscale input output
```

Upload or generate images into `input/` before running the command. You can display a result in a Python cell with:

```python
from PIL import Image
from IPython.display import display

display(Image.open("output/photo_gray.png"))
```

## Processing flow

1. `std::filesystem` walks the input directory and filters supported filename extensions.
2. `stbi_load` decodes each file and converts its pixels to three-channel RGB.
3. The program allocates input and output buffers on the GPU and copies the RGB pixels to the device.
4. `grayscaleKernel` assigns each pixel to a CUDA thread. A bounds check protects threads in the final, partially filled block.
5. The grayscale bytes are copied back to host memory and written as a one-channel PNG by `stbi_write_png`.

## Current scope and possible next steps

This is an educational first version, not yet a general-purpose image library. It handles one image at a time, uses basic per-image GPU allocations and transfers, and does not include performance benchmarking. Useful extensions include a CPU reference implementation, correctness checks, CUDA event timing, case-insensitive extension handling, and processing multiple images concurrently with streams.

## Third-party libraries

- [`stb_image`](https://github.com/nothings/stb) for image decoding.
- [`stb_image_write`](https://github.com/nothings/stb) for PNG output.

Include the upstream license/attribution notices for these headers in the repository, and check their current license terms before redistributing them.

## License

No project license is specified yet. Add a `LICENSE` file before presenting the project as open source; without one, others generally do not have an explicit license to reuse your code.
