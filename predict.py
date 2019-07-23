from rios import applier
from rios import fileinfo
from datetime import datetime
import numpy as np

date = 736582

def gen_prediction(info, infile, outfile, other_args):
    
    T = 365.25
    pi_val_simple = (2 * np.pi) / T
    pi_val_advanced = (4 * np.pi) / T
    pi_val_full = (6 * np.pi) / T
    
    num_input_bands = infile.coeff_img.shape[0]
    
    # Get number of bands
    num_output_bands = num_input_bands // 11
    
    # Set up array with the correct output shape
    px_out = np.zeros((num_output_bands, 45, 45), dtype='float64')
    
    # Each band is predicted separately
    for i in range(0, num_input_bands, 11):
        
        # Generate predicted values for this block, for this band
        prediction = (infile.coeff_img[i] * (date - infile.coeff_img[i+10])) + infile.coeff_img[i+1] + (infile.coeff_img[i+2] * np.cos(pi_val_simple*(date - infile.coeff_img[i+10]))) + (infile.coeff_img[i+3] * np.sin(pi_val_simple*(date - infile.coeff_img[i+10]))) + (infile.coeff_img[i+4] * np.cos(pi_val_advanced*(date - infile.coeff_img[i+10]))) + (infile.coeff_img[i+5] * np.sin(pi_val_advanced*(date - infile.coeff_img[i+10]))) + (infile.coeff_img[i+6] * np.cos(pi_val_full*(date - infile.coeff_img[i+10]))) + (infile.coeff_img[i+7] * np.sin(pi_val_full*(date - infile.coeff_img[i+10])))
        
        output_band = i // 11
        px_out[output_band] = prediction
    
    outfile.output_img = px_out
    
def predict_for_date(date, input_path, output_path, output_driver='KEA', num_processes=1):
    
    # Check/except input and output paths
    
    # Create object to hold input files    
    infile = applier.FilenameAssociations()
    infile.coeff_img = input_path
    
    # Create object to hold output file
    outfile = applier.FilenameAssociations()
    outfile.output_img = output_path
    
    # ApplierControls object holds details on how processing should be done
    app = applier.ApplierControls()
    
    # Set output file type
    app.setOutputDriverName(output_driver)
    
    # Use Python's multiprocessing module
    app.setJobManagerType('multiprocessing')
    app.setNumThreads(num_processes)
    
    # Open input file to get details
    coeff_img = fileinfo.ImageInfo(infile.coeff_img)
    
    # Get no data value from first layer
    nodata_val = coeff_img.nodataval[0]
    
    # Additional arguments - have to be passed as a single object
    other_args = applier.OtherInputs()
    other_args.date_to_predict = date
    
    applier.apply(gen_prediction, infile, outfile, otherArgs=other_args, controls=app)

predict_for_date('2019-01-01', 'output.kea', 'predicted.kea')