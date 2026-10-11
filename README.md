Panorama Stitching — Computer Vision

A Python application that combines multiple overlapping photographs into a single panorama using classical computer vision techniques.

About the Project

This project was developed as part of the Computer Vision  course. It uses feature detection, feature matching, homography estimation, and image blending to create panoramic images.

The application provides a graphical interface for selecting photographs and generating a panorama automatically.

Features

- Select between 2 and 10 photographs.
- Automatically resize large images.
- Detect image features using SIFT.
- Match features using BFMatcher and the Ratio Test.
- Estimate homography using RANSAC.
- Calculate transformations relative to a reference image.
- Align photographs using perspective warping.
- Blend overlapping areas using feather blending.
- Display processing statistics.
- Save the final panorama and feature-matching visualizations.

# Technologies

- Python
- OpenCV
- NumPy
- Tkinter

## Project Structure

```text
panorama_project/
├── images/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── panorama_stitching.py
├── requirements.txt
├── README.md
└── .gitignore
```

The `results/` directory is generated when the application runs and is excluded from version control.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/panorama-stitching.git
cd panorama-stitching
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment on Windows

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## How to Run

Run the application:

```bash
python panorama_stitching.py
```

1. Click the button to select photographs.
2. Choose between 2 and 10 images of the same scene.
3. Wait for feature detection and image matching.
4. View the generated panorama.
5. Find the output files in the `results/` directory.

## Recommended Input Images

For better results:

- Use photographs of the same scene taken from slightly different positions.
- Ensure neighboring photographs overlap by approximately 30–50%.
- Use images with sufficient lighting and visible details.
- Avoid excessive movement between photographs.

## Computer Vision Pipeline

1. **Image loading and resizing:** prepares photographs for processing.
2. **Feature extraction:** detects keypoints and descriptors using SIFT.
3. **Feature matching:** finds corresponding features using BFMatcher and the Ratio Test.
4. **Homography estimation:** uses RANSAC to reject unreliable matches.
5. **Image alignment:** calculates transformations relative to a reference photograph.
6. **Perspective warping:** maps photographs onto a common canvas.
7. **Feather blending:** combines overlapping areas using distance-based weights.
8. **Output generation:** saves and displays the resulting panorama.

## Output

- `panorama_final.jpg` — the final panoramic image.
- `matches_*.jpg` — visualizations of feature matches between photographs.

The application also reports the number of selected and used images, matches, RANSAC inliers, panorama dimensions, and processing time.

## Limitations

- Photographs must contain sufficient overlapping content.
- Large perspective changes and lighting differences may reduce image quality.
- Some photographs may be excluded if reliable matches cannot be found.
- The maximum number of input photographs is currently 10.

## Author

Academic project developed for the Computer Vision 1 course.
