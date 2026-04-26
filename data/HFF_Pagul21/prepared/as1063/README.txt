Cluster: abells1063
Field: AS1063clu
Lens redshift: 0.3480
Reference: RA=342.18550890, Dec=-44.53010010
Photometric rows: 2274
Magnification rows: 785
Pagul21 member rows: 300
BCG candidate id: 1215
SIMBAD sources queried: 457
SIMBAD photometry matches: 278
SIMBAD magnification matches: 179
obs_arcs rows using SIMBAD zspec: 80
Candidate families: 15
Candidate-family images: 34

WARNING:
The Pagul21 Zenodo files do not include multiple-image family membership.
The generated obs_arcs.cat has one image per catalog object and is intended
for parser/integration work only. Add real multiple-image family labels before
using this as a science strong-lensing fit.

Run parser/integration smoke test with:
python -m lenscluster.cluster_solver --par-path data/HFF_Pagul21/prepared/as1063/as1063_bootstrap.par --fit-mode joint --fit-method svi --svi-steps 1 --samples 2 --skip-validation --skip-plots
