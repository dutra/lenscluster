# Probabilistic Family Assignment

This note describes the current family partition engine. The important design choice is that final probability is placed on complete valid partitions, not on independent image labels or independent pair labels.

## Target

Let there be \(N\) observed images. A family assignment is a partition

$$
Z = \{z_1,\ldots,z_N\},
$$

where

$$
z_i = z_j
$$

means images \(i\) and \(j\) are multiple images of the same background source.

For a candidate family \(F_k = \{i:z_i=k\}\), the physical hypothesis is that all members share one latent source:

$$
H_{\rm same}(F_k):
\quad
\beta_i \approx \beta_k,\quad
z_i^{\rm src} \approx z_k^{\rm src},\quad
c_i \approx c_k.
$$

The alternative is that the images are unrelated catalog/background draws:

$$
H_{\rm bg}(F_k):
\quad
i\in F_k \text{ are independent sources}.
$$

The current final scoring model uses Bayes factors:

$$
s(Z)
=
\sum_{F_k\in Z}
\left[
\log {\rm BF}_{\rm lens}(F_k)
+
\log {\rm BF}_{\rm catalog}(F_k)
\right]
+
\log p(Z)
+
\log L_{\rm anchor}(Z).
$$

The normalized partition posterior over the generated candidate set is

$$
w_m
=
p(Z_m\mid D,\mathcal{C})
=
\frac{\exp(s(Z_m)/T)}
{\sum_{q=1}^{M}\exp(s(Z_q)/T)},
$$

where \(\mathcal{C}=\{Z_1,\ldots,Z_M\}\) is the candidate set and \(T\) is the partition temperature.

The probability-conserving pair probability is then derived from weighted full partitions:

$$
P_{ij}
=
P(z_i=z_j\mid D,\mathcal{C})
\approx
\sum_{m=1}^{M}
w_m\,\mathbf{1}\left[z_i^{(m)}=z_j^{(m)}\right].
$$

This is conserved because

$$
\sum_{m=1}^{M} w_m = 1.
$$

## Catalog Bayes Factor

The catalog Bayes factor compares whether the catalog data for a proposed family are better explained by one shared source or by independent background draws.

For a family \(F\),

$$
\log {\rm BF}_{\rm catalog}(F)
=
\log p(D_F^{\rm catalog}\mid H_{\rm same})
-
\log p(D_F^{\rm catalog}\mid H_{\rm bg}).
$$

The catalog data currently include:

- redshift or photo-z;
- colors from available magnitude columns;
- simple morphology such as size and ellipticity.

For a scalar observable \(y_i\), such as redshift or one color component, the same-source model is

$$
y_i \sim \mathcal{N}(\mu_F,\sigma_i^2+s^2),
$$

where \(\mu_F\) is the latent family value and \(s\) is extra intrinsic/catalog scatter.

The same-source marginal likelihood integrates over \(\mu_F\):

$$
p(y_F\mid H_{\rm same})
=
\int
p(\mu_F)
\prod_{i\in F}
\mathcal{N}(y_i\mid \mu_F,\sigma_i^2+s^2)
\,d\mu_F.
$$

The background model treats the members as independent draws from the catalog population:

$$
p(y_F\mid H_{\rm bg})
=
\prod_{i\in F}
\mathcal{N}(y_i\mid \mu_{\rm bg},\sigma_{\rm bg}^2).
$$

The full catalog score sums these likelihood ratios across redshift, colors, and morphology:

$$
\log {\rm BF}_{\rm catalog}(F)
=
w_z\log {\rm BF}_z(F)
+
w_c\log {\rm BF}_c(F)
+
w_m\log {\rm BF}_m(F).
$$

For the current clean mock tests, no outlier mixture is used.

## Lens Bayes Factor

The lens Bayes factor asks whether the proposed family ray-shoots to one shared source-plane location more strongly than expected from independent background source-plane positions.

For lens parameters \(\Theta\) and trial source redshift \(z\), image \(i\) ray-shoots to

$$
\hat{\boldsymbol{\beta}}_i(\Theta,z)
=
\mathbf{x}^{\rm obs}_i
-
\boldsymbol{\alpha}(\mathbf{x}^{\rm obs}_i;\Theta,z).
$$

The same-source model for a family \(F\) is

$$
\hat{\boldsymbol{\beta}}_i
\sim
\mathcal{N}
\left(
\boldsymbol{\beta}_F,
\Sigma_{\beta,i}
\right),
\qquad i\in F,
$$

where

$$
\Sigma_{\beta,i} = \sigma_i^2I + \sigma_{\rm src}^2I.
$$

In the implementation, \(\boldsymbol{\beta}_F\) is profiled by the precision-weighted source centroid:

$$
\hat{\boldsymbol{\beta}}_F
=
\frac{\sum_{i\in F} w_i\hat{\boldsymbol{\beta}}_i}
{\sum_{i\in F} w_i},
\qquad
w_i = \Sigma_{\beta,i}^{-1}.
$$

The same-source source-plane likelihood is

$$
\log p(\hat{\boldsymbol{\beta}}_F\mid H_{\rm same})
=
\sum_{i\in F}
\log
\mathcal{N}
\left(
\hat{\boldsymbol{\beta}}_i
\mid
\hat{\boldsymbol{\beta}}_F,
\Sigma_{\beta,i}
\right).
$$

The background model treats ray-shot source positions as independent draws from the source-plane population at that redshift:

$$
p(\hat{\boldsymbol{\beta}}_F\mid H_{\rm bg})
=
\prod_{i\in F}
p_{\rm bg}(\hat{\boldsymbol{\beta}}_i).
$$

Thus

$$
\log {\rm BF}_{\rm lens}(F)
=
\log p(\hat{\boldsymbol{\beta}}_F\mid H_{\rm same})
-
\log p(\hat{\boldsymbol{\beta}}_F\mid H_{\rm bg}).
$$

Because source redshift may be uncertain, the cached implementation evaluates a compact redshift grid and marginalizes:

$$
\log {\rm BF}_{\rm lens}(F)
=
\log
\sum_g
p(z_g\mid D_F^z)
\exp
\left[
\log {\rm BF}_{\rm lens}(F;z_g)
\right].
$$

The fast path precomputes:

- ray-shot source-plane positions for every image and redshift grid point;
- independent source-plane background log likelihoods;
- repeated family scores by member set.

## Partition Prior

The prior is currently CRP-like with size penalties:

$$
\log p(Z)
=
K\log\alpha
+
\log\Gamma(\alpha)
-
\log\Gamma(\alpha+N)
+
\sum_{k=1}^{K}\log\Gamma(n_k)
-
\lambda_1\sum_k \mathbf{1}(n_k=1)
-
\frac{1}{2}
\sum_{k:n_k\ge2}
\left(
\frac{n_k-n_0}{\sigma_n}
\right)^2.
$$

Definitions:

- \(K\): number of families.
- \(n_k\): size of family \(k\).
- \(\alpha\): concentration parameter.
- \(\lambda_1\): singleton penalty.
- \(n_0\): target multi-image family size.
- \(\sigma_n\): family-size tolerance.

This prior is still a calibration target. The shared-BF likelihood now carries most of the physical evidence.

## Anchor Penalty

High-confidence anchors are treated as strong constraints. Anchors can come from known same-family labels or incompatible redshift/color evidence.

For pair anchor \(a_{ij}\in\{0,1\}\),

$$
\log L_{\rm anchor}(Z)
=
-20
\sum_{(i,j)\in A}
w_{ij}
\mathbf{1}
\left[
\mathbf{1}(z_i=z_j)\ne a_{ij}
\right].
$$

## Candidate Generation

The final likelihood is family-level, but candidate generation still uses fast pair and lens affinities.

The engine builds candidate partitions from:

- thresholded connected components;
- complete-link and average-link clustering;
- greedy pivot correlation-clustering proposals;
- noisy affinity draws;
- source-plane clustering proposals;
- beam assembly of non-overlapping source-plane cluster hypotheses;
- local split, merge, and reassignment perturbations.

The pair logistic model remains useful for proposal generation, but it is no longer the preferred final scoring model. Pairwise likelihoods overcount correlated evidence, especially color and photo-z similarities, because a large family creates many pair terms from the same underlying information.

## Scorer-Guided Repair

Candidate generation can still miss an otherwise obvious partition if one image is left as a singleton or attached to the wrong family. The current engine therefore runs an optional scorer-guided local repair pass after the first final scoring pass.

For the top \(K\) scored partitions, it proposes:

- attach each singleton to each existing family;
- move each image to each existing family;
- merge each pair of families;
- split one image out of a family.

All repaired partitions are deduplicated and rescored with the same final shared-BF score:

$$
s(Z)
=
\sum_{F_k\in Z}
\left[
\log {\rm BF}_{\rm lens}(F_k)
+
\log {\rm BF}_{\rm catalog}(F_k)
\right]
+
\log p(Z)
+
\log L_{\rm anchor}(Z).
$$

Then weights are renormalized over the repaired candidate set.

This repair pass is not MCMC. It is a deterministic, scorer-guided expansion around high-probability candidates.

## Current Benchmark Behavior

On the 21-image mock with 5 blind families, shared BF scoring recovers the truth as the highest-weight partition:

```text
pair_score_mode: none
family_catalog_score_mode: likelihood_ratio
lens_source_score_mode: likelihood_ratio

ARI: 1.000
AUC: 1.000
AP: 1.000
truth rank: 1
```

On a larger 29-image mock with 7 blind families, shared BF scoring without repair was nearly correct but missed the exact truth candidate:

```text
ARI: 0.952
truth in candidates: false
```

Adding one repair round over the top 200 partitions recovered the truth:

```text
ARI: 1.000
AUC: 1.000
AP: 1.000
truth in candidates: true
truth rank: 1
```

The main remaining development target is calibration and speed for larger candidate sets, not replacing the shared-BF scoring structure.

## Lens-Model Uncertainty Pattern

Fully joint sampling of partitions and lens parameters,

$$
p(Z,\Theta\mid D),
$$

is not the recommended implementation target. The partition \(Z\) is discrete and combinatorial, while the lens parameters \(\Theta\) are continuous and expensive to evaluate. NUTS can sample \(\Theta\) conditional on a fixed partition, but it cannot sample the discrete partition itself. A custom split/merge/reassignment MCMC over \(Z\) jointly with \(\Theta\) would require difficult proposal design and would likely mix poorly.

The practical approximation is to separate lens-model uncertainty propagation from partition candidate search.

First fit the lens model using trusted high-confidence families:

$$
q(\Theta)
\approx
p(\Theta\mid D_{\rm trusted}).
$$

Then score candidate partitions under samples or representative draws from this lens posterior:

$$
\Theta^{(m)} \sim q(\Theta),
\qquad
m=1,\ldots,M_\Theta.
$$

For each lens draw, compute a shared-BF partition score:

$$
s(Z;\Theta^{(m)})
=
\sum_{F_k\in Z}
\left[
\log {\rm BF}_{\rm lens}(F_k;\Theta^{(m)})
+
\log {\rm BF}_{\rm catalog}(F_k)
\right]
+
\log p(Z)
+
\log L_{\rm anchor}(Z).
$$

Then marginalize partition evidence over lens uncertainty with log-sum-exp:

$$
s_{\rm marg}(Z)
=
\log
\frac{1}{M_\Theta}
\sum_{m=1}^{M_\Theta}
\exp\left[s(Z;\Theta^{(m)})\right].
$$

Use this marginalized score for the final partition weights:

$$
w_Z
=
\frac{\exp(s_{\rm marg}(Z)/T)}
{\sum_{Z'\in\mathcal{C}}\exp(s_{\rm marg}(Z')/T)}.
$$

The recommended workflow is:

1. Fit \(\Theta\) using trusted families.
2. Build a compact set of lens posterior draws or representative SVI/NUTS samples.
3. Generate candidate partitions using catalog and lens affinities.
4. Score candidates with catalog BF and lens BF for each \(\Theta^{(m)}\).
5. Marginalize scores over \(\Theta^{(m)}\).
6. Run scorer-guided local repair on top candidates.
7. Normalize partition weights over the repaired candidate set.
8. Optionally refit the lens model conditional on the top few partitions.

This is not exact joint inference, but it captures the important uncertainty propagation:

$$
p(Z\mid D)
\approx
\int
p(Z\mid D,\Theta)\,
p(\Theta\mid D_{\rm trusted})
\,d\Theta.
$$

It also keeps the difficult discrete search outside the continuous lens sampler.

## Practical Command

A representative validation command is:

```bash
python -m lenscluster.validation partition-benchmark \
  --run-name scoring_shared_bf_more_families_repair \
  --n-train-families 4 \
  --n-blind-families 7 \
  --min-images-per-family 2 \
  --partition-thresholds 8 \
  --partition-noisy-partitions 10 \
  --partition-temperature 2.0 \
  --partition-repair-top-k 200 \
  --partition-repair-rounds 1 \
  --pair-score-mode none \
  --family-catalog-score-weight 1.0 \
  --family-catalog-score-mode likelihood_ratio \
  --lens-source-plane-weight 1.0 \
  --lens-source-score-mode likelihood_ratio \
  --lens-source-redshift-grid 8
```
