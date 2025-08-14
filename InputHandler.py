import pandas as pd

def estimate_error(chain_outputs, column_name, burn_in = 0.3):
    
    # Remove burn-in samples and combine chains
    combined_data = []

    for i in range(len(chain_outputs)):
        df = pd.read_csv(chain_outputs[i], delim_whitespace=True)
        combined_data.append(df[column_name][int(len(df) * burn_in):])
    
    std = pd.concat(combined_data).std()

    return std

