import pandas as pd

def estimate_error(chain_outputs, column_name, burn_in = 0.3, array=False, max_index=None, file_type=None):
    """Estimate the standard deviation of a parameter from MCMC chain outputs.
    Parameters:
    ----------
    chain_outputs : 
        if array is False, str
            file path root containing MCMC chain data.
        if array is True, list of file paths containing MCMC chain data.
            List of DataFrames containing MCMC chain data.
    column_name : str
        The name of the column in the output files corresponding to the parameter.
    burn_in : float, optional
        Fraction of samples to discard as burn-in (default is 0.3).
    range : int, optional
        Required if array is False. Number of chain files to process. Refers to indexing.
    file_type : str, optional
        Type of the chain output files (e.g., 'csv', 'txt'). Needed if array is False.
    
    Returns:
    -------
    float
        Estimated standard deviation of the parameter.
    """

    # Remove burn-in samples and combine chains
    combined_data = []

    if array:
        for i in range(len(chain_outputs)):
            df = pd.read_csv(chain_outputs[i], sep='\s+')
            combined_data.append(df[column_name][int(len(df) * burn_in):])
    
    if not array:
        for i in range(max_index+1):
            df = pd.read_csv(f'{chain_outputs}{i}.{file_type}', sep='\s+')
            combined_data.append(df[column_name][int(len(df) * burn_in):])
    
    std = pd.concat(combined_data).std()

    return std

