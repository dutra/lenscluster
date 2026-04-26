Cluster: macs1149
Field: M1149clu
Lens redshift: 0.5430
Reference: RA=177.40024440, Dec=22.39696070
Photometric rows: 3630
Magnification rows: 1449
Pagul21 member rows: 300
BCG candidate id: 1372
SIMBAD sources queried: 1672
SIMBAD photometry matches: 1421
SIMBAD magnification matches: 676
obs_arcs rows using SIMBAD zspec: 221
Candidate families: 23
Candidate-family images: 57

WARNING:
The Pagul21 Zenodo files do not include multiple-image family membership.
The generated obs_arcs.cat has one image per catalog object and is intended
for parser/integration work only. Add real multiple-image family labels before
using this as a science strong-lensing fit.

Run parser/integration smoke test with:
python -m lenscluster.cluster_solver --par-path data/HFF_Pagul21/prepared/m1149/m1149_bootstrap.par --fit-mode joint --fit-method svi --svi-steps 1 --samples 2 --skip-validation --skip-plots
