Cluster: abell370
Field: A370clu
Lens redshift: 0.3750
Reference: RA=39.97281970, Dec=-1.57405795
Photometric rows: 2609
Magnification rows: 1312
Pagul21 member rows: 300
BCG candidate id: 1267
SIMBAD sources queried: 900
SIMBAD photometry matches: 552
SIMBAD magnification matches: 317
obs_arcs rows using SIMBAD zspec: 141
Candidate families: 20
Candidate-family images: 63

WARNING:
The Pagul21 Zenodo files do not include multiple-image family membership.
The generated obs_arcs.cat has one image per catalog object and is intended
for parser/integration work only. Add real multiple-image family labels before
using this as a science strong-lensing fit.

Run parser/integration smoke test with:
python -m lenscluster.cluster_solver --par-path data/HFF_Pagul21/prepared/a370/a370_bootstrap.par --fit-mode joint --fit-method svi --svi-steps 1 --samples 2 --skip-validation --skip-plots
