This repository contains a set of functions and classes allowing you to use RIOS to process a stack of raster images into a single output image containing per-band season-trend model coefficients, RMSE, and an overall value per-band. The outputs and model fitting are based on the following paper:

Zhu, Z.; Woodcock, C.E.; Holden, C.; Yang, Z. Generating synthetic Landsat images based on all available Landsat data: Predicting Landsat surface reflectance at any given time. *Remote Sensing of Environment* **2015**, *162*, 67–83. doi:10.1016/j.rse.2015.02.009.

Models are fitted over the entire provided time series, i.e. the script does not look for breaks/changes.

The input is a JSON file with a list of date:filepath pairs as strings, e.g:

{"YYYY-MM-DD": "/path/to/image/file/1.tif",
"YYYY-MM-DD": "/path/to/image/file/2.tif",
"YYYY-MM-DD": "/path/to/image/file/3.tif"}

See also the included file example.json. The files do not have to be listed in order of date.

There is also a function for using the generated models to predict a new image for any given date, which takes the first image as input.
