# fiber-bundle
(c) Stefan Hiemer, 2025

This code was used to provide the results of the paper "Fiber bundle model of thermally activated creep failure" by S. Hiemer et al.. (DOI follows once published on Arxiv).

If you want to recreate the figures from the paper, download the associated h5file from Zenodo (10.5281/zenodo.16413747) and execute the scripts named "fig*.py".

If you want to perform some fiber bundle simulations on your own, check the file run.py and the function submit_jobs. If exectued, this generates an HDF5 file, from which information can be extracted via the function read_h5.py. If you want to recreate the figures from the paper with the new data, please make sure that the loads, temperatures etc. are adapted in the functions according to the provided doc strings.

The Python packages used to generate and postprocess the data can be found in requirements.txt. The python version was Python 3.12.10.
