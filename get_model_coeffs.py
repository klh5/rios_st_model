import json
from rios import applier
from rios import fileinfo
from datetime import datetime
from makemodel import MakeSeasonTrendModel
from multiprocessing import Value, Lock
import numpy as np

class PercentDone(object):
    
    """Class for storing the percent complete. Safe to access from multiple
    processes."""
    
    def __init__(self):
        self.val = Value('i', -1)
        self.lock = Lock()

    def set_percent(self, percent):
        with self.lock:
            self.val.value = percent

    def get_percent(self):
        with self.lock:
            return self.val.value

def gen_per_band_models(info, inputs, outputs, other_args):
    
    """This function is run per-block by RIOS. In this case each block is a 
    single pixel. Given a block of values for each band for each date, returns
    a numpy array containing the model coefficients, RMSE, and an overall 
    value for each band."""
    
    nodata_val = other_args.nodata_val
    
    num_bands = other_args.num_bands
    
    progress_tracker = other_args.progress_tracker
    
    # Calculate number of outputs
    num_outputs = num_bands * 9
    
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
            st_model = MakeSeasonTrendModel(masked_dates)
                
            # Fit Lasso model. Complexity will be determined by number of observations
            st_model.fit_model(masked, False)
            
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
    
    curr_percent = info.getPercent()
    
    # Update progress tracker. RIOS only returns an integer so checking the current value
    # ensures that this only prints when the percentage changes. Has to be safe for access
    # across all processes
    if(curr_percent > progress_tracker.get_percent()):
        progress_tracker.set_percent(curr_percent)
        print('{}%'.format(curr_percent))
    
    outputs.outimage = px_out
    
def gen_layer_names(bands):
    
    """Given a list of band numbers, returns a list of layer names. These
    make it easier to identify which values are which in the output image."""
    
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
    
def get_ST_model_coeffs(json_fp, output_fp, output_driver='KEA', bands=None, num_processes=1):
    
    """Main function to run to generate the output image. Given an input JSON file
    and an output file path, generates a multi-band output image where each pixel
    contains the model details for that pixel. Opening/closing of files, generation
    of blocks and use of multiprocessing is all handled by RIOS."""
    
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
    layer_names = gen_layer_names(bands)
    app.setLayerNames(layer_names)
    
    # Additional arguments - have to be passed as a single object
    other_args = applier.OtherInputs()
    other_args.dates = dates
    other_args.num_bands = num_bands
    other_args.nodata_val = nodata_val
    
    progress_tracker = PercentDone()
    other_args.progress_tracker = progress_tracker
    
    applier.apply(gen_per_band_models, infiles, outfiles, otherArgs=other_args, controls=app)
    
get_ST_model_coeffs('example.json', 'test_output.kea', bands=[3,4,5,6,7])

