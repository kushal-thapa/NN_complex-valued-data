# NN_complex-valued-data
## Performance comparison of time-series, magnitude spectrogram and rectangular spectrogram datasets in a simple Neural Network

* Test1.py script a) creates artificial time-series data of ASK, FSK and PSK signals b) creates magnitude and rectangular spectrograms from the time-series data c) uses time-series, magnitude spectrogram, and rectangular spectrogram dataset in a simple Neural Network

* Test1_data_file_N-10 is a data file resulting from Test1.py (with num_tests=10). Each row of this data file has 8 comma seperated values corresponding to:
['Type of modulation','SNR','Type of dataset (time-series (0), magnitude spectrogram (1) or rectangular spectrogram (2))','Test accuracy of NN model','Test loss of NN model','Precision of NN model','Recall of NN model','F1 of NN model']

* An example python code (based on pandas) to evaluate this data file is shown in data_processing_script.py
