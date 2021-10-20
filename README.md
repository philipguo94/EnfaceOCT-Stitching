# Rota_stitching
The repository is to auto stitching ROTA macular and ROTA Optic images automatically. 
It includes:
1) a sperate UNet deep learning model to extract the vessel;
2) Stitch the images according to vessel extracted.

### Usage

Change the path inside the main.py and run main.py

the results will be saved in "result" folder and the error IDs will be saved as .npy.

### Some issues to know
The code environment is
    
    python 3.7
    opencv-python 4.2.0.34
    numpy 1.18.1
    mstplotlib 3.1.3
    scikit-image 3.1.3
    scipy 1.4.1

