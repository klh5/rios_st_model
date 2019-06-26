import json
from rios import applier
from rios import fileinfo
from datetime import datetime
from makeModel import MakeSeasonTrendModel
import numpy as np

def genPerBandModels(info, inputs, outputs, other_args):
    
    nodata_val = other_args.nodata_val
    
    num_bands = other_args.num_bands
    
    # Get length of time series
    len_ts = len(inputs.images)
    
    # Calculate number of outputs
    num_outputs = num_bands * 9
    
    # Set up array with the correct output shape
    px_out = np.zeros((num_outputs, 1, 1), dtype='float64')
    
    # Keep track of which layer to write to in the output file
    layer = 0
    
    # Get data for one band at a time
    for band in range(0, num_bands):
        
        band_data = np.array([inputs.images[t][band][0][0] for t in range(0, len_ts)])
        
        # Get indices of missing values
        mask = np.where(band_data == nodata_val)
        
        # Drop missing data points from band data
        masked = np.delete(band_data, mask)
        
        # Check if any data is left once no data values have been removed
        if masked.size > 6:
        
            # Drop missing data points from dates
            masked_dates = np.delete(other_args.dates, mask)
            
            # Initialise model class
            st_model = MakeSeasonTrendModel(masked_dates, len_ts)
                
            # Fit Lasso model. Complexity will be determined by number of observations
            st_model.fitModel(masked, False)
            
            # Extract coefficients for output
            coeffs = st_model.coefficients # Slope, cos1, sin1, cos2, sin2, cos3, sin3
            slope = coeffs[0]
            px_out[layer][0][0] = slope
            
            px_out[layer+1] = st_model.RMSE
            
            # Get middle date
            mid_ts = (masked_dates[-1] - masked_dates[0]) / 2
                 
            # Calculate overall value for period
            intercept = st_model.lasso_model.intercept_
            overall_val = intercept + (slope * mid_ts)
                 
            px_out[layer+2][0][0] = overall_val
            
            # Pad out coefficients
            # Some models might not have second or third harmonic terms - these are set to 0 to allow the same classifier
            # to be used
            coeffs = np.pad(coeffs, (0, 7-len(coeffs)), 'constant')
            
            # Add harmonic coefficients to output
            px_out[layer+3][0][0] = coeffs[1]
            px_out[layer+4][0][0] = coeffs[2]
            px_out[layer+5][0][0] = coeffs[3]
            px_out[layer+6][0][0] = coeffs[4]
            px_out[layer+7][0][0] = coeffs[5]
            px_out[layer+8][0][0] = coeffs[6]
        
        layer += 9 # There are always 9 outputs per band
    
    outputs.outimage = px_out
    
def genLayerNames(bands):
    
    layer_names = []
    
    for band in bands:
        
        layer_names.append('band{}_slope'.format(band))
        layer_names.append('band{}_RMSE'.format(band))
        layer_names.append('band{}_overall'.format(band))
        layer_names.append('band{}_cos1'.format(band))
        layer_names.append('band{}_sin1'.format(band))
        layer_names.append('band{}_cos2'.format(band))
        layer_names.append('band{}_sin2'.format(band))
        layer_names.append('band{}_cos3'.format(band))
        layer_names.append('band{}_sin3'.format(band))
        
    return(layer_names)
    
def getSTModelCoeffs(json_fp, output_fp, output_driver='KEA', bands=None, num_processes=1):
    
    paths = []
    dates = []
    
    # Open and read JSON file containing date:filepath pairs
    with open(json_fp) as json_file:  
        image_list = json.load(json_file)
    
        for date, img_path in image_list.items():
            dates.append(datetime.strptime(date, '%Y-%m-%d').toordinal())
            paths.append(img_path)
    
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
        
    template_image = None
        
    # Set up output layer names based on band numbers
    layer_names = genLayerNames(bands)
    app.setLayerNames(layer_names)
    
    # Additional arguments - have to be passed as a single object
    other_args = applier.OtherInputs()
    other_args.dates = dates
    other_args.num_bands = num_bands
    other_args.nodata_val = nodata_val
    
    applier.apply(genPerBandModels, infiles, outfiles, otherArgs=other_args, controls=app)
    
getSTModelCoeffs('example_full.json', 'test_output.kea', bands=[3,4,5,6,7])

