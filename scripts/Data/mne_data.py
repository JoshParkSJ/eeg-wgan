import mne

def download_EEGBCI(subject=0, runs=[1], path='/', verbose=False):
    """
    Downloads EEG BCI data from mne.datasets.eegbci
    The url this data comes from is: https://physionet.org/files/eegmmidb/1.0.0/

    Parameters
    ----------
    subject : [int]
        The subject number to download data for
    runs : [int]
        The run number to download data for
    path : str
        The path to download the data to
    verbose : bool
        Whether to print out all the details or not
    """
    if type(subject) == int:
        mne.datasets.eegbci.load_data(subject, runs, path, verbose=verbose)   
    elif type(subject) == list:
        for sub in subject:
            mne.datasets.eegbci.load_data(sub, runs, path, verbose=verbose)
    
    print("Data downloaded!")