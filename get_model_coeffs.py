import json
import sys
from rios import applier
from rios import fileinfo
from datetime import datetime
from makemodel import MakeSeasonTrendModel
import numpy as np

def gen_per_band_models(info, inputs, outputs, other_args):
    
    """Run per-block by RIOS. In this case each block is a 
    single pixel. Given a block of values for each band for each date, returns
    a numpy array containing the model coefficients, RMSE, and an overall 
    value for each band."""
    
    nodata_val = other_args.nodata_val
    
    num_bands = other_args.num_bands
    
    # Calculate number of outputs
    num_outputs = num_bands * 11
    
    # Set up array with the correct output shape
    px_out = np.zeros((num_outputs, 1, 1), dtype='float64')
    
    # Keep track of which layer to write to in the output file
    layer = 0
    
    # Get data for one band at a time
    for band in range(0, num_bands):
        
        band_data = np.array([inputs.images[t][band][0][0] for t in range(0, len(inputs.images))])
        
        # Get indices of missing values
        mask = np.where(band_data == nodata_val)
        
        # Drop missing data points from band data
        masked = np.delete(band_data, mask)
        
        # Check if any data is left once no data values have been removed
        if masked.size >= 6:
        
            # Drop missing data points from dates
            masked_dates = np.delete(other_args.dates, mask)
            
            # Initialise model class
            st_model = MakeSeasonTrendModel(masked_dates, masked)
            
            # If Lasso model is chosen, fit the model using the variables provided
            # By default, alpha=1 and cross validation is not done
            if(other_args.model_type == 'Lasso'):
                st_model.fit_lasso_model(other_args.cv, other_args.alpha)
                
            else:
                st_model.fit_ols_model()
            
            # Extract coefficients for output
            coeffs = st_model.coefficients # Slope, cos1, sin1, cos2, sin2, cos3, sin3
            slope = coeffs[0]
            intercept = st_model.model.intercept_
            px_out[layer] = slope
            px_out[layer+1] = intercept
            
            # Pad out coefficients
            # Some models might not have second or third harmonic terms - these are set to 0 to allow the same classifier
            # to be used
            coeffs = np.pad(coeffs, (0, 7-len(coeffs)), 'constant')
            
            # Add harmonic coefficients to output
            px_out[layer+2] = coeffs[1]
            px_out[layer+3] = coeffs[2]
            px_out[layer+4] = coeffs[3]
            px_out[layer+5] = coeffs[4]
            px_out[layer+6] = coeffs[5]
            px_out[layer+7] = coeffs[6]
            
            px_out[layer+8] = st_model.RMSE
            
            # Get middle date
            mid_ts = (masked_dates[-1] - masked_dates[0]) / 2
                 
            # Calculate overall value for period
            intercept = st_model.model.intercept_
            overall_val = intercept + (slope * mid_ts)
                 
            px_out[layer+9][0][0] = overall_val
            
            px_out[layer+10] = st_model.start_date
            
        layer += 11 # There are always 11 outputs per band
    
    curr_percent = float(info.yblock * info.xtotalblocks + info.xblock) / float(info.xtotalblocks * info.ytotalblocks) * 100
    print('{:.2f}'.format(curr_percent))
    
    outputs.outimage = px_out
    
def gen_layer_names(bands):
    
    """Given a list of band numbers, returns a list of layer names. These
    make it easier to identify which values are which in the output image."""
    
    layer_names = []
    
    for band in bands:
        
        layer_names.append('{}_slope'.format(band))
        layer_names.append('{}_intercept'.format(band))
        layer_names.append('{}_cos1'.format(band))
        layer_names.append('{}_sin1'.format(band))
        layer_names.append('{}_cos2'.format(band))
        layer_names.append('{}_sin2'.format(band))
        layer_names.append('{}_cos3'.format(band))
        layer_names.append('{}_sin3'.format(band))
        layer_names.append('{}_RMSE'.format(band))
        layer_names.append('{}_overall'.format(band))
        layer_names.append('{}_start'.format(band))
        
    return(layer_names)
    
def get_ST_model_coeffs(json_fp, output_fp, output_driver='KEA', bands=None, num_processes=1, model_type='Lasso', alpha=20, cv=False):
    
    """Main function to run to generate the output image. Given an input JSON file
    and an output file path, generates a multi-band output image where each pixel
    contains the model details for that pixel. Opening/closing of files, generation
    of blocks and use of multiprocessing is all handled by RIOS.
    
    json_fp:       Path to JSON file of date/filepath pairs.
    output_fp:     Path for output file.
    output_driver: Short driver name for GDAL, e.g. KEA, GTiff.
    bands:         List of GDAL band numbers to use in the analysis, e.g. [2, 5, 7].
    num_processes: Number of concurrent processes to use.
    model_type:    Either 'Lasso' or 'OLS'. The type of model fitting to use. OLS will 
                   be faster, but more likely to overfit. Both types will adjust the number of model
                   coefficients depending on the number of observations.
    alpha:         If using Lasso fitting, the alpha value controls the degree of
                   penalization of the coefficients. The lower the value, the closer 
                   the model will fit the data. For surface reflectance, a value of 
                   around 20 (the default) is usually OK.
    cv:            If using Lasso fitting, you can use cross validation to choose
                   the value of alpha by setting cv=True. However, this is not recommended and will
                   substantially increase run time."""
    
    paths = []
    dates = []
    
    try:
        # Open and read JSON file containing date:filepath pairs
        with open(json_fp) as json_file:  
            image_list = json.load(json_file)
        
            for date, img_path in image_list.items():
                dates.append(datetime.strptime(date, '%Y-%m-%d').toordinal())
                paths.append(img_path)
    except FileNotFoundError:
        print('Could not find the provided JSON file.')
        sys.exit()
    except json.decoder.JSONDecodeError as e:
        print('There is an error in the provided JSON file: {}'.format(e))
        sys.exit()
        
    # Create object to hold input files    
    infiles = applier.FilenameAssociations()
    infiles.images = paths
    
    # Create object to hold output file
    outfiles = applier.FilenameAssociations()
    outfiles.outimage = output_fp
    
    # ApplierControls object holds details on how processing should be done
    app = applier.ApplierControls()
    
    # Set window size to 1 because we are working per-pixel
    app.setWindowXsize(1)
    app.setWindowYsize(1)
    
    # Set output file type
    app.setOutputDriverName(output_driver)
    
    # Use Python's multiprocessing module
    app.setJobManagerType('multiprocessing')
    app.setNumThreads(num_processes)
    
    # Open first image in list to use as a template
    template_image = fileinfo.ImageInfo(infiles.images[0])
    
    # Get no data value
    nodata_val = template_image.nodataval[0]
    
    if not bands: # No bands specified - default to all
        
        num_bands = template_image.RasterCount
        bands = [i for i in range(1, num_bands+1)]
    
    else: # If a list of bands is provided
    
        # Number of bands determines things like the size of the output array
        num_bands = len(bands)
    
        # Need to tell the applier to only use the specified bands 
        app.selectInputImageLayers(bands)
    
    # Create list of actual names
    
    full_names = [template_image.layerNameFromNumber(i) for i in bands]    
    template_image = None
        
    # Set up output layer names based on band numbers
    layer_names = gen_layer_names(full_names)
    app.setLayerNames(layer_names)
    
    # Additional arguments - have to be passed as a single object
    other_args = applier.OtherInputs()
    other_args.dates = dates
    other_args.num_bands = num_bands
    other_args.nodata_val = nodata_val
    other_args.model_type = model_type
    other_args.alpha = alpha
    other_args.cv = cv
    
    try:
        applier.apply(gen_per_band_models, infiles, outfiles, otherArgs=other_args, controls=app)
    except RuntimeError as e:
        print('There was an error processing the images: {}'.format(e))
        print('Do all images in the JSON file exist?')

# Example    
get_ST_model_coeffs('example.json', 'output.kea', bands=[3,4,5,6,7], num_processes=4)

