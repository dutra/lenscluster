Cluster: abell2744
Field: A2744clu
Lens redshift: 0.3080
Reference: RA=3.58771205, Dec=-30.39960590
Photometric rows: 2766
Magnification rows: 1304
Pagul21 member rows: 300
BCG candidate id: 1200
SIMBAD sources queried: 4275
SIMBAD photometry matches: 2584
SIMBAD magnification matches: 1194
obs_arcs rows using SIMBAD zspec: 364
Candidate families: 32
Candidate-family images: 91

WARNING:
The Pagul21 Zenodo files do not include multiple-image family membership.
The generated obs_arcs.cat has one image per catalog object and is intended
for parser/integration work only. Add real multiple-image family labels before
using this as a science strong-lensing fit.

Run parser/integration smoke test with:
python -m lenscluster.cluster_solver --par-path data/HFF_Pagul21/prepared/a2744/a2744_bootstrap.par --fit-mode joint --fit-method svi --svi-steps 1 --samples 2 --skip-validation --skip-plots
