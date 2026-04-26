Cluster: macs0416
Field: M0416clu
Lens redshift: 0.3960
Reference: RA=64.03626750, Dec=-24.07504930
Photometric rows: 2536
Magnification rows: 977
Pagul21 member rows: 300
BCG candidate id: 932
SIMBAD sources queried: 4187
SIMBAD photometry matches: 2433
SIMBAD magnification matches: 914
obs_arcs rows using SIMBAD zspec: 285
Candidate families: 35
Candidate-family images: 91

WARNING:
The Pagul21 Zenodo files do not include multiple-image family membership.
The generated obs_arcs.cat has one image per catalog object and is intended
for parser/integration work only. Add real multiple-image family labels before
using this as a science strong-lensing fit.

Run parser/integration smoke test with:
python -m lenscluster.cluster_solver --par-path data/HFF_Pagul21/prepared/m0416/m0416_bootstrap.par --fit-mode joint --fit-method svi --svi-steps 1 --samples 2 --skip-validation --skip-plots
