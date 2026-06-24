# Critical-Arc-Aware Image-Plane Likelihood

This note explains a simple way to use one likelihood for both ordinary
point-like multiple images and extended critical arcs. The main idea is to
measure image residuals in the local directions set by the lensing Jacobian.

## The Problem

For a lens model, an image-plane position

$$
\boldsymbol{\theta} = (\theta_x, \theta_y)
$$

maps to a source-plane position

$$
\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta}),
$$

where $\boldsymbol{\alpha}$ is the lens deflection. Images from the same source
family should map back to the same source position.

A common likelihood penalizes the distance between the predicted and observed
image positions as if the uncertainty were the same in every image-plane
direction:

$$
\log p_i =
-\frac{1}{2}
\left[
\frac{|\boldsymbol{r}_i|^2}{\sigma_i^2}
+ \log \sigma_i^2
\right].
$$

Here $\boldsymbol{r}_i$ is an image-plane residual and $\sigma_i$ is the
astrometric uncertainty for image $i$.

This is reasonable for ordinary compact images. It is less appropriate for
critical arcs. Near a critical curve, one direction in the image plane is highly
stretched. The arc can slide along this stretched direction without strongly
changing the source-plane position. Therefore, the data constrain the
cross-arc direction much more strongly than the along-arc direction.

## Local Lens Coordinates

The local lensing Jacobian is

$$
A_i =
\frac{\partial \boldsymbol{\beta}}{\partial \boldsymbol{\theta}}
\bigg|_{\boldsymbol{\theta}_i}.
$$

Near a critical curve, this matrix has one small singular value. Informally,
this means that one image-plane direction maps only weakly back to the source
plane. That direction is the stretched or critical direction.

We can define two local image-plane directions:

- $\hat{\boldsymbol{e}}_{\parallel,i}$: the stretched direction, roughly along
  the arc;
- $\hat{\boldsymbol{e}}_{\perp,i}$: the well-constrained direction, roughly
  across the arc.

These directions define two projection matrices:

$$
P_{\parallel,i}
=
\hat{\boldsymbol{e}}_{\parallel,i}
\hat{\boldsymbol{e}}_{\parallel,i}^{\mathsf{T}},
$$

and

$$
P_{\perp,i}
=
I - P_{\parallel,i}.
$$

Any residual can then be split into an along-arc part and a cross-arc part:

$$
r_{\parallel,i}^2
=
\boldsymbol{r}_i^{\mathsf{T}}
P_{\parallel,i}
\boldsymbol{r}_i,
$$

$$
r_{\perp,i}^2
=
\boldsymbol{r}_i^{\mathsf{T}}
P_{\perp,i}
\boldsymbol{r}_i.
$$

## A Unified Likelihood for Points and Arcs

Instead of using the same uncertainty in every direction, use an anisotropic
image-plane covariance:

$$
\Sigma_i
=
\sigma_{\perp,i}^2 P_{\perp,i}
+
\sigma_{\parallel,i}^2 P_{\parallel,i}.
$$

The likelihood is then

$$
\log p_i
=
-\frac{1}{2}
\left[
\boldsymbol{r}_i^{\mathsf{T}}
\Sigma_i^{-1}
\boldsymbol{r}_i
+
\log |\Sigma_i|
\right].
$$

Equivalently, in the local lens coordinates,

$$
\log p_i
=
-\frac{1}{2}
\left[
\frac{r_{\perp,i}^2}{\sigma_{\perp,i}^2}
+
\frac{r_{\parallel,i}^2}{\sigma_{\parallel,i}^2}
+
\log \sigma_{\perp,i}^2
+
\log \sigma_{\parallel,i}^2
\right].
$$

This likelihood handles both ordinary point images and critical arcs.

For an ordinary point-like image far from a critical curve,

$$
\sigma_{\parallel,i} \approx \sigma_{\perp,i}.
$$

Then the likelihood reduces to the usual isotropic image-position likelihood.

For a critical arc,

$$
\sigma_{\parallel,i} \gg \sigma_{\perp,i}.
$$

Then the model strongly penalizes residuals across the arc but only weakly
penalizes residuals along the arc. This matches the physical information in an
arc: the cross-arc position is usually much more informative than the exact
position along the stretched direction.

## Choosing the Along-Arc Uncertainty

The key modeling choice is how to set $\sigma_{\parallel,i}$.

Let $s_{\min,i}$ be the smaller singular value of the local lensing Jacobian
$A_i$. Near a critical curve,

$$
s_{\min,i} \rightarrow 0.
$$

A simple smooth model is

$$
\sigma_{\parallel,i}^2
=
\sigma_i^2
+
\sigma_{\rm arc}^2 g(s_{\min,i}),
$$

where $g(s_{\min,i})$ is close to 0 away from a critical curve and close to 1
near a critical curve.

A more lensing-aware model uses the fact that small source-plane uncertainty
maps into large image-plane uncertainty near a critical curve:

$$
\sigma_{\parallel,i}^2
=
\sigma_i^2
+
\min \left[
\left(\frac{\sigma_i}{s_{\rm eff,i}}\right)^2,
\sigma_{\max}^2
\right],
$$

with

$$
s_{\rm eff,i}
=
\sqrt{s_{\min,i}^2 + \epsilon^2}.
$$

The small number $\epsilon$ prevents division by zero and keeps gradients
finite. The cap $\sigma_{\max}$ prevents the along-arc uncertainty from becoming
unphysically large.

## Why Stabilization Is Still Needed

The anisotropic likelihood removes the bad assumption that arcs should be
treated like compact point images. It does not remove the mathematical
singularity in the lens mapping.

Near a critical curve, the inverse Jacobian behaves roughly like

$$
A_i^{-1} \sim \frac{1}{s_{\min,i}}.
$$

Derivatives of quantities involving $s_{\min,i}$ can become very large. This can
make SVI or NUTS unstable, even if the likelihood is scientifically reasonable.

For that reason, practical implementations usually include:

- a smooth floor on $s_{\min,i}$;
- a cap on the maximum along-arc variance;
- smooth transition functions instead of hard thresholds;
- sometimes stopped gradients through purely geometric gates.

These choices make the likelihood a stable approximation instead of an exact
singular mathematical object.

## Relation to the Current Lenscluster Critical-Arc Likelihood

The current critical-arc likelihood in `lenscluster` is already close to this
coordinate-transform idea. It uses the local Jacobian to identify the critical
direction, projects residuals into critical and non-critical components, and
inflates the covariance along the critical direction.

The implemented version adds several practical layers:

- a damped inverse or Levenberg-Marquardt-like step to convert source-plane
  residuals into image-plane residuals;
- a mixture between point-like and arc-like behavior;
- reliability and robust outlier terms;
- image-presence penalties;
- covariance caps and singular-value stabilization.

So the clean anisotropic likelihood is the conceptual core. The implementation
adds robustness and sampler-stability details needed for real fitting.

## Lenscluster Option

`lenscluster` now exposes this cleaner anisotropic version as:

```python
stage1_likelihood = "critical-arc-anisotropic"
```

The same value can be used for `stage0_likelihood` and `stage2_forward_mode`
when those stages should use the anisotropic critical-arc likelihood. Internally
this maps to:

```text
critical-arc-anisotropic-image-plane
```

The existing option

```python
stage1_likelihood = "critical-arc"
```

continues to select the older robust mixture implementation,
`critical-arc-mixture-image-plane`.

## Image-Plane Versus Source-Plane Use

This unified likelihood is most natural in the image plane. The anisotropic
covariance is an image-plane statement:

$$
\Sigma_{\theta,i}
=
\sigma_{\perp,i}^2 P_{\perp,i}
+
\sigma_{\parallel,i}^2 P_{\parallel,i}.
$$

It directly says that the cross-arc image direction is strongly constrained,
while the along-arc image direction is weakly constrained. In image-plane mode,
the likelihood is

$$
\log p_i
=
-\frac{1}{2}
\left[
\boldsymbol{r}_{\theta,i}^{\mathsf{T}}
\Sigma_{\theta,i}^{-1}
\boldsymbol{r}_{\theta,i}
+
\log |\Sigma_{\theta,i}|
\right].
$$

The same idea can be used in a source-plane approximation, but it must be
translated through the local lensing Jacobian. If

$$
A_i =
\frac{\partial \boldsymbol{\beta}}{\partial \boldsymbol{\theta}},
$$

then the corresponding source-plane covariance is

$$
\Sigma_{\beta,i}
=
A_i \Sigma_{\theta,i} A_i^{\mathsf{T}}.
$$

The source-plane residual likelihood would then be

$$
\log p_i
=
-\frac{1}{2}
\left[
\boldsymbol{r}_{\beta,i}^{\mathsf{T}}
\Sigma_{\beta,i}^{-1}
\boldsymbol{r}_{\beta,i}
+
\log |\Sigma_{\beta,i}|
\right].
$$

This is possible, but it is less clean near critical curves. The stretched
image-plane direction maps to a very small source-plane direction, so the
source-plane covariance can become numerically delicate. It can also become
less transparent scientifically, because the actual observed data are image
positions and arc shapes, not source-plane positions.

The practical recommendation is:

- use the unified anisotropic likelihood directly in image-plane mode;
- use it in local-Jacobian image-plane mode when a faster approximation is
  needed;
- treat pure source-plane use as a fast preconditioning approximation, not the
  final critical-arc likelihood;
- keep the current critical-arc mode as a robust implementation of this same
  core idea, with extra mixture and stabilization terms.

## Scientific Interpretation

This likelihood should be described as a critical-arc-aware image-plane
surrogate likelihood. It is more physically appropriate than an isotropic
point-image likelihood near critical curves, because it respects the fact that
arcs constrain different directions with different strength.

It is not a full pixel-level generative model of an arc. It does not model the
surface brightness, PSF, source morphology, or detection process. Therefore, its
hyperparameters should be calibrated and validated on simulations.

The main validation tests should be:

- Does it improve image residuals compared with an isotropic point likelihood?
- Does it recover the correct convergence or mass profile in mocks?
- Does it avoid biasing lens parameters near critical curves?
- Are the recovered uncertainties calibrated?

If these tests pass, the likelihood is a scientifically useful approximation
for catalog-level strong-lensing inference.
