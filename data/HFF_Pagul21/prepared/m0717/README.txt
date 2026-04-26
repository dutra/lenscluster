Cluster: macs0717
Field: M0717clu
Lens redshift: 0.5450
Reference: RA=109.39393270, Dec=37.74445580
Photometric rows: 2912
Magnification rows: 853
Pagul21 member rows: 300
BCG candidate id: 2001
SIMBAD sources queried: 1865
SIMBAD photometry matches: 1275
SIMBAD magnification matches: 379
obs_arcs rows using SIMBAD zspec: 74
Candidate families: 6
Candidate-family images: 16

WARNING:
The Pagul21 Zenodo files do not include multiple-image family membership.
The generated obs_arcs.cat has one image per catalog object and is intended
for parser/integration work only. Add real multiple-image family labels before
using this as a science strong-lensing fit.

Run parser/integration smoke test with:
python -m lenscluster.cluster_solver --par-path data/HFF_Pagul21/prepared/m0717/m0717_bootstrap.par --fit-mode joint --fit-method svi --svi-steps 1 --samples 2 --skip-validation --skip-plots
