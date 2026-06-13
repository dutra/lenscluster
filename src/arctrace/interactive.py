"""Interactive matplotlib session: click a seed, review the fit, label, accept.

One field panel with the fit drawn on top of the image. Workflow:

  click an arc       measure the arc at the click (the fit is overlaid in place)
  drag               pan the image
  scroll wheel       zoom the field view in/out toward the cursor
  enter in label box accept the pending measurement under that label
  r                  reject the pending measurement
  + / -              raise/lower the segmentation threshold and re-measure
  g / G              lower/raise the gap-bridging scale and re-measure
  u                  undo the last accepted arc
  q                  finish: write arcfile, side table, QA PNGs

Every accept is appended to <output>/session_autosave.json for crash safety.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from skimage import measure as sk_measure

from . import geometry
from .arcfile import ArcfileWriteError, write_arcfile, write_sidetable
from .config import ArcMeasureConfig
from .measure import ArcMeasurement, BandArtifacts, measure_arc
from .mosaics import BandMosaic, mosaic_center
from .qa import arc_summary_lines, extract_display_cutouts, render_display_image, save_qa_png
from .regions import normalize_arc_id

LOGGER = logging.getLogger("arctrace")


# Headless backends; note that the interactive *Agg backends (TkAgg, QtAgg,
# GTK3Agg, ...) also contain "agg", so membership must be an exact match.
_NON_INTERACTIVE_BACKENDS = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}


def _select_gui_backend(preferred: str | None = None) -> str:
    """Activate a working interactive matplotlib backend before pyplot is imported.

    Honors an explicit ``preferred`` (from ``--backend``) or ``MPLBACKEND`` first,
    then falls back to a priority order. On Wayland, Qt's window often fails to map,
    so TkAgg (rendered via XWayland) is preferred there.
    """
    import os

    import matplotlib

    override = preferred or os.environ.get("MPLBACKEND")
    if override and override.lower() in _NON_INTERACTIVE_BACKENDS:
        raise RuntimeError(
            f"Requested backend {override!r} is headless; interactive mode needs a "
            "GUI backend such as TkAgg or QtAgg."
        )

    wayland = os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
    default_order = ["TkAgg", "QtAgg", "GTK3Agg"] if wayland else ["QtAgg", "TkAgg", "GTK3Agg"]
    candidates: list[str] = []
    for name in ([override] if override else []) + default_order:
        if name and name not in candidates:
            candidates.append(name)

    tried: list[str] = []
    for name in candidates:
        try:
            matplotlib.use(name, force=True)
            return name
        except (ImportError, ValueError) as exc:
            tried.append(f"{name} ({exc.__class__.__name__})")
    raise RuntimeError(
        "No interactive matplotlib backend could be activated (tried: "
        + ", ".join(tried)
        + "). Install tkinter or PyQt, or pass --backend with a working GUI backend."
    )


class _Session:
    def __init__(
        self,
        mosaics: dict[str, BandMosaic],
        cfg: ArcMeasureConfig,
        *,
        center: SkyCoord,
        view_size_arcsec: float,
        output_dir: Path,
        reliability_override: float | None,
        arcfile_name: str,
        overwrite: bool,
        no_qa: bool,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import TextBox

        self.mosaics = mosaics
        self.cfg = cfg
        self.output_dir = output_dir
        self.reliability_override = reliability_override
        self.arcfile_name = arcfile_name
        self.overwrite = overwrite
        self.no_qa = no_qa
        self.accepted: list[ArcMeasurement] = []
        self.accepted_artifacts: list[BandArtifacts] = []
        self.pending: ArcMeasurement | None = None
        self.pending_artifacts: BandArtifacts | None = None
        self.pending_seed: SkyCoord | None = None
        self.used_labels: set[str] = set()
        self.finished = False
        self.autosave_path = output_dir / "session_autosave.json"
        self._overlay_artists: list = []
        self._pan: dict | None = None
        self._dragged = False

        self.fig = plt.figure(figsize=(10.5, 9.5))
        grid = self.fig.add_gridspec(2, 1, height_ratios=[20, 1])
        self.ax_field = self.fig.add_subplot(grid[0, 0])
        ax_box = self.fig.add_subplot(grid[1, 0])
        self.textbox = TextBox(ax_box, "arc ID (enter=accept) ", initial="")
        self.textbox.on_submit(self._on_label_submit)

        self.field_cutouts = extract_display_cutouts(mosaics, center, view_size_arcsec)
        reference_cutout = self.field_cutouts.get(cfg.reference_band)
        if reference_cutout is None:
            raise RuntimeError(f"Could not extract a field view in {cfg.reference_band}.")
        self.field_wcs = reference_cutout.wcs
        image, band_desc, is_rgb = render_display_image(self.field_cutouts, cfg.reference_band)
        self.ax_field.imshow(image, origin="lower", cmap=None if is_rgb else "gray", interpolation="nearest")
        self._field_title = f"field [{band_desc}]"
        self.ax_field.set_xticks([])
        self.ax_field.set_yticks([])
        self._set_status("click an arc; drag to pan, scroll to zoom")

        self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)

    # ------------------------------------------------------------------ events
    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax_field or event.xdata is None or event.ydata is None:
            return
        # Scroll up zooms in, down zooms out, keeping the cursor point fixed.
        scale = 1.0 / 1.2 if event.button == "up" else 1.2
        xlim = self.ax_field.get_xlim()
        ylim = self.ax_field.get_ylim()
        relx = (event.xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (event.ydata - ylim[0]) / (ylim[1] - ylim[0])
        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        self.ax_field.set_xlim(event.xdata - new_w * relx, event.xdata + new_w * (1.0 - relx))
        self.ax_field.set_ylim(event.ydata - new_h * rely, event.ydata + new_h * (1.0 - rely))
        self.fig.canvas.draw_idle()

    def _on_press(self, event) -> None:
        if event.inaxes is not self.ax_field or event.button != 1:
            return
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # toolbar pan/zoom active
        # Capture the original limits and transform so the pan stays drift-free.
        self._pan = {
            "press": (event.x, event.y),
            "xlim": self.ax_field.get_xlim(),
            "ylim": self.ax_field.get_ylim(),
            "inv": self.ax_field.transData.inverted(),
            "xdata": event.xdata,
            "ydata": event.ydata,
        }
        self._dragged = False

    def _on_motion(self, event) -> None:
        if self._pan is None or event.x is None or event.y is None:
            return
        px, py = self._pan["press"]
        if not self._dragged and math.hypot(event.x - px, event.y - py) < 3.0:
            return  # below the drag threshold; still a click
        self._dragged = True
        d0 = self._pan["inv"].transform((px, py))
        d1 = self._pan["inv"].transform((event.x, event.y))
        dx, dy = d0[0] - d1[0], d0[1] - d1[1]
        xlim, ylim = self._pan["xlim"], self._pan["ylim"]
        self.ax_field.set_xlim(xlim[0] + dx, xlim[1] + dx)
        self.ax_field.set_ylim(ylim[0] + dy, ylim[1] + dy)
        self.fig.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self._pan is None:
            return
        pan, self._pan = self._pan, None
        if self._dragged:
            return  # it was a pan, not a click
        xdata, ydata = pan["xdata"], pan["ydata"]
        if xdata is None or ydata is None:
            return
        world = self.field_wcs.pixel_to_world(float(xdata), float(ydata))
        self._measure_at(SkyCoord(ra=world.ra, dec=world.dec, frame="icrs"))

    def _on_key(self, event) -> None:
        if event.key == "q":
            self.finished = True
            import matplotlib.pyplot as plt

            plt.close(self.fig)
        elif event.key == "r":
            self.pending = None
            self.pending_artifacts = None
            self.pending_seed = None
            self._set_status("rejected; click another arc")
        elif event.key in {"+", "="}:
            self._adjust(threshold_factor=1.0 / 1.1)
        elif event.key == "-":
            self._adjust(threshold_factor=1.1)
        elif event.key == "g":
            self._adjust(gap_factor=0.5)
        elif event.key == "G":
            self._adjust(gap_factor=2.0)
        elif event.key == "u":
            self._undo()

    def _on_label_submit(self, text: str) -> None:
        if self.pending is None:
            return
        text = text.strip()
        if not text:
            return
        try:
            label = normalize_arc_id(text)
        except ValueError as exc:
            LOGGER.error("%s", exc)
            return
        if label in self.used_labels:
            LOGGER.error("Label %s already used in this session.", label)
            return
        measurement = replace(self.pending, label=label)
        self.accepted.append(measurement)
        if self.pending_artifacts is not None:
            self.accepted_artifacts.append(self.pending_artifacts)
        self.used_labels.add(label)
        self._autosave()
        LOGGER.info("Accepted %s (%d arcs so far).", label, len(self.accepted))
        self.pending = None
        self.pending_artifacts = None
        self.pending_seed = None
        self.textbox.set_val("")
        self._set_status(f"accepted {label}; click the next arc")

    # ------------------------------------------------------------- measurement
    def _measure_at(self, seed: SkyCoord, cfg: ArcMeasureConfig | None = None) -> None:
        cfg = cfg or self.cfg
        measurement, artifacts = measure_arc(
            self.mosaics,
            seed,
            cfg,
            label=None,
            reliability_override=self.reliability_override,
            return_artifacts=True,
        )
        self.pending_seed = seed
        if not measurement.success:
            self.pending = None
            self.pending_artifacts = None
            self._set_status(f"measurement failed: {measurement.failure_reason}")
            LOGGER.warning("Measurement failed: %s", measurement.failure_reason)
            return
        self.pending = measurement
        self.pending_artifacts = artifacts.get(measurement.reference_band)
        self.cfg = cfg  # keep any threshold/gap adjustments for subsequent arcs
        if self.pending_artifacts is not None:
            self._draw_arc_overlay(measurement, self.pending_artifacts)
            self.ax_field.set_title(f"{self._field_title} — review fit; label+enter to accept, r to reject")
            self.fig.canvas.draw_idle()
        else:
            self._set_status("measured, but no overlay available for the reference band")
        LOGGER.info(
            "phi=%.4f+-%.4f rad, kappa=%.4f+-%.4f /arcsec; type a label and press enter to accept.",
            measurement.tangent_angle_offset_rad,
            measurement.sigma_tangent_rad,
            measurement.curvature_arcsec_inv,
            measurement.sigma_curvature_arcsec_inv,
        )

    def _adjust(self, *, threshold_factor: float = 1.0, gap_factor: float = 1.0) -> None:
        if self.pending_seed is None:
            return
        segmentation = replace(
            self.cfg.segmentation,
            threshold_sigma=self.cfg.segmentation.threshold_sigma * threshold_factor,
            max_bridge_gap_arcsec=self.cfg.segmentation.max_bridge_gap_arcsec * gap_factor,
        )
        cfg = replace(self.cfg, segmentation=segmentation)
        LOGGER.info(
            "Re-measuring with threshold %.2f sigma, gap %.2f arcsec.",
            segmentation.threshold_sigma,
            segmentation.max_bridge_gap_arcsec,
        )
        self._measure_at(self.pending_seed, cfg)

    def _undo(self) -> None:
        if not self.accepted:
            return
        removed = self.accepted.pop()
        if self.accepted_artifacts:
            self.accepted_artifacts.pop()
        if removed.label:
            self.used_labels.discard(removed.label)
        self._autosave()
        LOGGER.info("Removed %s.", removed.label)
        self._set_status(f"removed {removed.label}; {len(self.accepted)} arcs remain")

    # ------------------------------------------------------------------ overlay
    def _clear_overlay(self) -> None:
        while self._overlay_artists:
            artist = self._overlay_artists.pop()
            try:
                artist.remove()
            except (ValueError, NotImplementedError):
                pass

    def _set_status(self, message: str) -> None:
        self._clear_overlay()
        self.ax_field.set_title(f"{self._field_title} — {message}")
        self.fig.canvas.draw_idle()

    def _draw_arc_overlay(self, measurement: ArcMeasurement, artifacts: BandArtifacts) -> None:
        """Draw the mask, ridge, fitted circle and tangent on the field image, in
        field-pixel coordinates (every overlay goes offsets-frame -> field WCS)."""
        self._clear_overlay()
        cutout = artifacts.cutout
        ra0, dec0 = artifacts.ra0_deg, artifacts.dec0_deg
        fwcs = self.field_wcs
        ax = self.ax_field
        artists = self._overlay_artists

        # Mask boundary (cutout pixels -> offsets -> field pixels).
        for contour in sk_measure.find_contours(artifacts.segmentation.mask.astype(float), 0.5):
            xo, yo = geometry.pixel_points_to_offset_frame(cutout.wcs, contour[:, 1], contour[:, 0], ra0, dec0)
            cx, cy = geometry.offset_frame_point_to_pixel(fwcs, xo, yo, ra0, dec0)
            artists.extend(ax.plot(cx, cy, color="#00e5ff", linewidth=0.9, alpha=0.9))

        # Ridge points, sized by SNR.
        ridge = artifacts.ridge
        xo, yo = geometry.pixel_points_to_offset_frame(cutout.wcs, ridge.x_pix, ridge.y_pix, ra0, dec0)
        rx, ry = geometry.offset_frame_point_to_pixel(fwcs, xo, yo, ra0, dec0)
        snr = np.array([p.snr for p in ridge.points])
        sizes = 8.0 + 20.0 * np.clip(snr / max(snr.max(), 1.0), 0.0, 1.0)
        artists.append(ax.scatter(rx, ry, s=sizes, facecolors="none", edgecolors="#ffd54f", linewidths=0.9))

        # Fitted circle.
        fit = artifacts.fit
        if not fit.is_line and math.isfinite(fit.r):
            theta = np.linspace(0.0, 2.0 * math.pi, 720)
            circle_x = fit.xc + fit.r * np.cos(theta)
            circle_y = fit.yc + fit.r * np.sin(theta)
            px, py = geometry.offset_frame_point_to_pixel(fwcs, circle_x, circle_y, ra0, dec0)
            artists.extend(ax.plot(px, py, color="#ff5252", linewidth=1.2))

        # Tangent segment at the anchor, anchor marker, and seed marker.
        anchor_x, anchor_y = artifacts.anchor_offsets
        phi = measurement.tangent_angle_offset_rad
        if math.isfinite(phi):
            half = 1.2  # arcsec
            seg_x = np.array([anchor_x - half * math.cos(phi), anchor_x + half * math.cos(phi)])
            seg_y = np.array([anchor_y - half * math.sin(phi), anchor_y + half * math.sin(phi)])
            px, py = geometry.offset_frame_point_to_pixel(fwcs, seg_x, seg_y, ra0, dec0)
            artists.extend(ax.plot(px, py, color="#69f0ae", linewidth=2.0))
        apx, apy = geometry.offset_frame_point_to_pixel(fwcs, anchor_x, anchor_y, ra0, dec0)
        artists.extend(ax.plot(float(apx), float(apy), marker="+", color="#69f0ae", markersize=12, markeredgewidth=2.0))
        spx, spy = geometry.offset_frame_point_to_pixel(fwcs, 0.0, 0.0, ra0, dec0)
        artists.extend(ax.plot(float(spx), float(spy), marker="x", color="#ff4081", markersize=10, markeredgewidth=1.6))

        # Summary text box.
        artists.append(
            ax.text(
                0.02,
                0.98,
                "\n".join(arc_summary_lines(measurement)),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.55, "boxstyle": "round,pad=0.35"},
            )
        )

    def _autosave(self) -> None:
        payload = []
        for measurement in self.accepted:
            record = dataclasses.asdict(measurement)
            record.pop("config_summary", None)
            payload.append(record)
        self.autosave_path.write_text(json.dumps(payload, indent=2, default=str))

    def finalize(self) -> list[ArcMeasurement]:
        if self.accepted:
            try:
                write_arcfile(self.accepted, self.output_dir / self.arcfile_name, overwrite=self.overwrite)
                LOGGER.info("Wrote %d arcs to %s.", len(self.accepted), self.output_dir / self.arcfile_name)
            except ArcfileWriteError as exc:
                LOGGER.error("Arcfile not written: %s", exc)
            write_sidetable(
                self.accepted,
                self.output_dir / "arctrace_sidetable.csv",
                self.output_dir / "arctrace_sidetable.json",
            )
            if not self.no_qa:
                for measurement, artifacts in zip(self.accepted, self.accepted_artifacts):
                    safe_label = (measurement.label or "arc").replace(".", "_")
                    save_qa_png(
                        measurement,
                        artifacts,
                        self.output_dir / "qa" / f"{safe_label}.png",
                        self.field_cutouts,
                    )
        else:
            LOGGER.info("No arcs accepted; nothing written.")
        return self.accepted


def run_interactive_session(
    mosaics: dict[str, BandMosaic],
    cfg: ArcMeasureConfig,
    *,
    center: SkyCoord | None,
    view_size_arcsec: float,
    output_dir: Path,
    reliability_override: float | None = None,
    arcfile_name: str = "arcfile.cat",
    overwrite: bool = False,
    no_qa: bool = False,
    backend: str | None = None,
) -> list[ArcMeasurement] | None:
    selected_backend = _select_gui_backend(backend)
    LOGGER.info("Using matplotlib backend %s.", selected_backend)
    import matplotlib.pyplot as plt

    if center is None:
        center = mosaic_center(mosaics[cfg.reference_band])
    session = _Session(
        mosaics,
        cfg,
        center=center,
        view_size_arcsec=view_size_arcsec,
        output_dir=output_dir,
        reliability_override=reliability_override,
        arcfile_name=arcfile_name,
        overwrite=overwrite,
        no_qa=no_qa,
    )
    print(__doc__)
    plt.show(block=True)
    return session.finalize()
