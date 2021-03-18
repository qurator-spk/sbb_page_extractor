# sbb_page_extractor
This tool allows you to extract page ( border or print space) from a image document. This tool can also extract the same page for a corresponding image of 
the main image (for example label of main image).

## Installation

      ./make.sh

## How to use 
    
      page_extractor -i <image file name> -o <output file name> -m <directory of model> -ci <corresponding image file name> -co <corresponding output file name>
      
      
## Model
Model can be found here https://qurator-data.de/sbb_textline_detector/. The name of the model is 'model_page_mixed_best'.
