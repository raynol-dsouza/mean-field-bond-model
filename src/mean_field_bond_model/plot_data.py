"""
Publication-ready 3x2 summary figure for one unary system (e.g. Fe, Mg).

Column 1 (rows 1-3)
    xy-slice bond densities for shells 1, 2, 3 (MD filled contours with MFBM
    line contours overlaid, as in plot_bond_density_slices). All three share
    the same delta-x axis, so only the bottom panel carries x tick labels and
    the x label. A single shared horizontal colorbar spans column 1, drawn just
    above it (ticks only, no title).
Column 2 (rows 1-3)
    (row 1) NVT + NPT anharmonic energy vs T, (row 2) NVT pressure vs T [GPa],
    (row 3) NPT lattice expansion vs T [%]. A shared legend spans column 2, in
    the same top strip as the colorbar.

Every panel is forced to the same square box size (ax.set_box_aspect(1)), and
each panel's caption is drawn inline at its "(a)" tag instead of as a separate
axes title, so rows can be packed with almost no vertical gap between them.

The helpers below mirror the conventions of mean_field_bond_model.plot
(font sizes, bond labels) as self-contained copies.

Usage
-----
    from mean_field_bond_model.plot_data import plot_unary_summary

    fig, axes = plot_unary_summary(
        disps=disps,
        shells=[
            {"wyckoff_site": 0, "shell": 1, "rho_mfbm": rho1, "rho_md": md1},
            {"wyckoff_site": 0, "shell": 2, "rho_mfbm": rho2, "rho_md": md2},
            {"wyckoff_site": 0, "shell": 3, "rho_mfbm": rho3, "rho_md": md3},
        ],
        symmetry_analysis=symmetry_analysis,
        nvt={
            "T": T_nvt,
            "U_ah_mfb": U_ah_nvt_mfb,
            "U_ah_md": U_ah_nvt_md,
            "U_ah_md_err": U_ah_nvt_md_err,
            "P_mfb": P_nvt_mfb,
            "P_md": P_nvt_md,
        },
        npt={
            "T": T_npt,
            "U_ah_mfb": U_ah_npt_mfb,
            "U_ah_md": U_ah_npt_md,
            "U_ah_md_err": U_ah_npt_md_err,
            "eps_mfb": eps_npt_mfb,
            "eps_md": eps_npt_md,
        },
    )
    fig.savefig("mg_summary.pdf")
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# First two colors of the seaborn "colorblind" palette.
COLORS = ["#0173b2", "#de8f05"]

__all__ = ["plot_unary_summary"]


# Tuned for rendering directly at a single journal column (~3.4-3.6 in wide,
# the default `figsize`) with NO further shrinking in LaTeX -- i.e.
# \includegraphics[width=\columnwidth] with no scale factor. If you render
# wider and let LaTeX shrink it, scale these fontsizes up by the same factor
# you'll shrink by, or the print copy will end up too small.
DEFAULT_SUMMARY_FONTSIZES = {
    "caption": 6.5,
    "label": 7,
    "tick": 6.5,
    "legend": 7,
    "clabel": 6.5,
}


def _apply_fontsizes(ax, fontsizes):
    ax.xaxis.label.set_fontsize(fontsizes["label"])
    ax.yaxis.label.set_fontsize(fontsizes["label"])
    ax.tick_params(axis="both", labelsize=fontsizes["tick"])


def _bond_species_label(symmetry_analysis, wyckoff_site, shell):
    """Same convention as plot.py's private helper, but abbreviating shell ->
    s, e.g. 'Mg-Mg, s = 1'."""
    wyckoff_sites = symmetry_analysis["Wyckoff_sites"]
    basis_sites = symmetry_analysis["basis_sites"]
    wyckoff_data = wyckoff_sites[wyckoff_site]
    origin_species = wyckoff_data["species"]
    representative_basis_site = wyckoff_data["basis_sites"][0]
    target_species = basis_sites[representative_basis_site]["shells"][shell][
        "target_species"
    ]
    return f"{origin_species}-{target_species}, s = {shell}"


def _nvt_npt_legend_style():
    """
    Fixed color/marker/linestyle convention shared by every thermo panel: color
    carries the ensemble (NVT red, NPT blue), linestyle/marker carries the
    source (MFB solid o, MD dashed x) -- learned once from the shared legend,
    then reused silently in the pressure and lattice-expansion panels (each of
    which only shows one ensemble/color).
    """
    return {
        "nvt_mfb": dict(
            color=COLORS[0],
            linestyle="-",
            marker="o",
            ms=3,
            lw=1.5,
            markerfacecolor="none",
        ),
        "nvt_md": dict(color="k", linestyle=":", marker="x", ms=3.5, lw=1),
        "npt_mfb": dict(
            color=COLORS[1],
            linestyle="-",
            marker="o",
            ms=3,
            lw=1.5,
            markerfacecolor="none",
        ),
        "npt_md": dict(color="k", linestyle="--", marker="x", ms=3.5, lw=1),
    }


def _close_row_gaps(fig, axes, n_stabilize=10):
    """
    box_aspect(1) shrinks each panel to fit the narrower of its cell's
    width/height, which left slack in the taller dimension -- visible as blank
    space between rows even with hspace=0 and h_pad=0, since those only set the
    *requested* spacing and constrained_layout still centers the shrunk box
    within its cell.

    Rescaling the whole figure to close that gap sounds simple but isn't:
    constrained_layout is an iterative solver that needs several draws to
    converge at a *fixed* size (confirmed empirically -- positions were still
    drifting after 3-4 redraws), and resizing on top of a not-yet-converged
    measurement compounds into a runaway feedback loop rather than converging.
    So instead: let the automatic layout fully converge first (repeated draws
    at fixed size), freeze it (`set_layout_engine("none")` so nothing
    re-adjusts these axes on the next draw), then directly stack each column's
    rows 1 and 2 flush under row 0 using row 0's own converged box size. That's
    a one-shot, deterministic placement rather than an approximate rescale, so
    it doesn't depend on this call's particular label widths.
    """
    for _ in range(n_stabilize):
        fig.canvas.draw()
    fig.set_layout_engine("none")
    for c in range(axes.shape[1]):
        p0 = axes[0, c].get_position()
        for r in range(1, axes.shape[0]):
            pr = axes[r, c].get_position()
            axes[r, c].set_position(
                [pr.x0, p0.y0 - r * p0.height, pr.width, p0.height]
            )


def _trim_outer_whitespace(fig):
    """
    _close_row_gaps packs the *rows* tightly, but leaves the figure's own
    canvas at whatever `figsize` was passed in -- so all the height that
    packing freed up shows up as blank space below the content (and any
    leftover margin constrained_layout reserved on the other three sides stays
    too). `bbox_inches="tight"` on savefig hides this, but only for that one
    save call; a plain fig.savefig(...) or a notebook's inline display still
    shows the full untrimmed canvas.

    Fix it once, on the actual Figure: measure the true tight bounding box of
    everything drawn (fig.get_tightbbox, the same measurement
    bbox_inches="tight" uses internally), shift every axes so that box's corner
    sits at the figure origin, and shrink the figure to that box's size. After
    this, the figure *is* its content -- any save or display path shows the
    same tightly-cropped result with no extra step required.
    """
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())  # in inches
    W, H = fig.get_size_inches()
    new_W = bbox.width
    new_H = bbox.height
    dx = -bbox.x0
    dy = -bbox.y0
    for ax in fig.axes:
        pos = ax.get_position()
        x0_in = pos.x0 * W + dx
        y0_in = pos.y0 * H + dy
        w_in = pos.width * W
        h_in = pos.height * H
        ax.set_position(
            [x0_in / new_W, y0_in / new_H, w_in / new_W, h_in / new_H]
        )
    fig.set_size_inches(new_W, new_H, forward=True)


def _close_column_gap(fig, axes, cbar_ax, legend_ax, target_gap_in=0.04):
    """
    _close_row_gaps only tightens vertical spacing; the horizontal gap between
    column 0 (densities + colorbar) and column 1 (properties + legend) is
    untouched by that, and is mostly column 1's own reserved space for its
    y-axis label/ticks, plus whatever w_pad/ wspace constrained_layout added on
    both sides of the gap.

    Fix analogous to _close_row_gaps: measure the true rendered right edge of
    column 0's content and left edge of column 1's content (via get_tightbbox,
    which -- unlike get_position -- includes tick labels and axis labels, i.e.
    what's actually visible), then shift the whole of column 1 left so only
    `target_gap_in` remains between them. Must run after _close_row_gaps has
    frozen the layout, and before _trim_outer_whitespace (which will then
    absorb the extra space this frees up on the right edge of the figure).
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    dpi = fig.dpi

    col0_right_in = max(a.get_tightbbox(renderer).x1 for a in axes[:, 0]) / dpi
    col1_left_in = min(a.get_tightbbox(renderer).x0 for a in axes[:, 1]) / dpi
    shift_in = (col1_left_in - col0_right_in) - target_gap_in
    if shift_in <= 0:
        return

    for ax in list(axes[:, 1]) + [legend_ax]:
        pos = ax.get_position()
        ax.set_position([pos.x0 - shift_in / W, pos.y0, pos.width, pos.height])


def plot_unary_summary(
    disps: np.ndarray,
    shells: list,
    symmetry_analysis: dict,
    nvt: dict,
    npt: dict,
    n_levels: int = 6,
    figsize: tuple = (3.6, 5.4),
    fontsizes: dict | None = None,
    density_xlim: tuple | None = None,
    density_ylim: tuple | None = None,
    show_panel_labels: bool = True,
    savepath: str | None = None,
):
    """
    3x2 results-section summary figure for one unary system: bond densities
    (shells 1-3) in column 1, thermodynamic properties (NVT+NPT anharmonic
    energy, NVT pressure, NPT lattice expansion) in column 2. All six panels
    share one square box size; the colorbar (column 1) and legend (column 2)
    occupy a matching thin strip on top.

    Parameters
    ----------
    disps : np.ndarray, 1D
        Shared displacement axis for every shell's bond-density grid
        (Angstrom).
    shells : list of 3 dict
        One dict per shell, in the order they should appear (rows 1-3 of
        column 1), with keys "wyckoff_site" (int), "shell" (int), "rho_mfbm"
        (np.ndarray, shape (nx, ny, nz)) and "rho_md" (same shape, or None).
        rho_md may be omitted (None) to plot MFBM alone (viridis fill, no
        overlay) for that shell; mixing shells with and without MD is fine.
    symmetry_analysis : dict
        As returned by NeighborBondSymmetryAnalysis.analyze(); used only for
        the inline captions of the density panels.
    nvt : dict
        "T" (K), "U_ah_mfb" (anharmonic energy), "U_ah_md" (or None),
        "U_ah_md_err" (or None; plotted as a shaded band around U_ah_md),
        "P_mfb" (bar) and "P_md" (bar, or None).
    npt : dict
        "T" (K), "U_ah_mfb", "U_ah_md" (or None), "U_ah_md_err" (or None),
        "eps_mfb" (isotropic strain, 1.0 = unstrained) and "eps_md" (or None).
    n_levels : int, optional
        Contour levels for the bond-density panels (shared across all three
        shells).
    figsize : tuple, optional
        Defaults to (3.6, 5.4) in -- a standard single PRB/APS journal column,
        meant to be dropped into LaTeX at
        \\includegraphics[width=\\columnwidth] with NO extra scale factor. This
        is only a *starting* height: the function measures the actual rendered
        row spacing for this call's data/labels and rescales height
        automatically so rows pack with zero gap (see `_close_row_gaps`) --
        width is left untouched, so the column-width fit is preserved
        regardless. fontsizes are tuned to be legible at this physical width.
    fontsizes : dict, optional
        Overrides for any of DEFAULT_SUMMARY_FONTSIZES's keys.
    density_xlim, density_ylim : tuple, optional
        Zoom for the three bond-density panels. Defaults to (disps.min(),
        disps.max()).
    show_panel_labels : bool, optional
        Defaults to True: prefix each inline caption with an (a)-(f) tag,
        standard for journal multi-panel figures and handy for referring to
        panels from the caption.
    savepath : str, optional
        If given, the figure is saved here (format inferred from extension) in
        addition to being returned.

    Returns
    -------
    fig, axes
        axes has shape (3, 2); column 0 = densities, column 1 = properties.
    """
    fontsizes = {**DEFAULT_SUMMARY_FONTSIZES, **(fontsizes or {})}
    if len(shells) != 3:
        raise ValueError(
            "shells must contain exactly 3 entries (shell 1, 2, 3 panels)."
        )

    panel_labels = iter("abcdef")

    def _caption(ax, text):
        """Inline, non-bold panel caption drawn at the same spot the (a)-(f)
        tag lives, with a soft white backing so it stays legible over filled
        contours/curves.
        """
        label = f"({next(panel_labels)}) {text}" if show_panel_labels else text
        ax.text(
            0.03,
            0.94,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=fontsizes["caption"],
            fontweight="normal",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1),
        )

    fig = plt.figure(figsize=figsize, layout="constrained")
    # row 0: a thin strip hosting the horizontal colorbar (col 0) and the
    # legend (col 1), at matching height; rows 1-3: density (col 0) / property
    # (col 1) pairs.
    gs = fig.add_gridspec(
        4, 2, height_ratios=[0.06, 1.0, 1.0, 1.0], hspace=0.0, wspace=0.0
    )
    # zero out constrained_layout's own padding engine too -- hspace=0 above
    # only sets the gridspec's requested spacing, but the layout engine adds
    # its own h_pad on top of that unless told not to.
    fig.get_layout_engine().set(h_pad=0.0, hspace=0.0)

    axes = np.empty((3, 2), dtype=object)
    for r in range(3):
        axes[r, 0] = fig.add_subplot(gs[r + 1, 0])
        axes[r, 1] = fig.add_subplot(gs[r + 1, 1])
        for c in range(2):
            axes[r, c].set_box_aspect(
                1
            )  # force every panel to the same square box size
            axes[r, c].set_anchor(
                "N"
            )  # pin to the top of the cell (no vertical centering gap)
    cbar_ax = fig.add_subplot(gs[0, 0])
    legend_ax = fig.add_subplot(gs[0, 1])
    legend_ax.axis("off")

    # ---------------------------------------------------------------- column
    # 1: bond densities
    iz = len(disps) // 2
    xlim = density_xlim or (float(disps.min()), float(disps.max()))
    ylim = density_ylim or (float(disps.min()), float(disps.max()))

    normalized = []
    for s in shells:
        rho_mfbm = s["rho_mfbm"]
        slice_mfbm = rho_mfbm[:, :, iz]
        slice_mfbm = slice_mfbm / slice_mfbm.sum()
        rho_md = s.get("rho_md")
        has_md = rho_md is not None
        slice_md = None
        if has_md:
            slice_md = rho_md[:, :, iz]
            slice_md = slice_md / slice_md.sum()
        normalized.append((slice_mfbm, slice_md, has_md))

    filled_arrays = [(sm if hm else sf) for sf, sm, hm in normalized]
    global_min = min(a.min() for a in filled_arrays)
    global_max = max(a.max() for a in filled_arrays)
    shared_levels = np.linspace(global_min, global_max, n_levels + 1)

    H, V = np.meshgrid(disps, disps, indexing="ij")
    contourf_mappable = None
    for row, (s, (slice_mfbm, slice_md, has_md)) in enumerate(
        zip(shells, normalized)
    ):
        ax = axes[row, 0]
        if has_md:
            cf = ax.contourf(
                H, V, slice_md, levels=shared_levels, cmap="coolwarm"
            )
            ax.contour(
                H,
                V,
                slice_mfbm,
                levels=shared_levels,
                cmap="inferno",
                linewidths=1.0,
            )
        else:
            cf = ax.contourf(
                H, V, slice_mfbm, levels=shared_levels, cmap="viridis"
            )
        if contourf_mappable is None:
            contourf_mappable = cf

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.set_ylabel(r"$\delta y\ [\mathrm{\AA}]$", labelpad=0)
        if row == 2:
            # shared delta-x axis: only the bottom density panel gets x
            # ticks/label
            ax.set_xlabel(r"$\delta x\ [\mathrm{\AA}]$", labelpad=0)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        _apply_fontsizes(ax, fontsizes)
        _caption(
            ax,
            _bond_species_label(
                symmetry_analysis, s["wyckoff_site"], s["shell"]
            ),
        )

    cbar = fig.colorbar(
        contourf_mappable,
        cax=cbar_ax,
        orientation="horizontal",
        ticks=shared_levels[::2],
        format="%.0e",
    )
    cbar_ax.xaxis.set_ticks_position("top")
    cbar.ax.tick_params(labelsize=fontsizes["clabel"], length=2, pad=0)

    # mark each MFBM contour line's level directly on the colorbar, colored the
    # same way the contour lines themselves are (inferno, normalized over the
    # same level range) -- this is what replaces the inline clabel() numbers:
    # the level is readable off the colorbar instead of cluttering each panel.
    contour_norm = plt.Normalize(vmin=shared_levels[0], vmax=shared_levels[-1])
    for lvl in shared_levels:
        cbar_ax.axvline(
            lvl, color=plt.cm.inferno(contour_norm(lvl)), linewidth=1.1
        )

    # ---------------------------------------------------------------- column
    # 2: thermodynamics
    style = _nvt_npt_legend_style()

    # (row 1) combined NVT + NPT anharmonic energy
    ax = axes[0, 1]
    T_nvt = np.asarray(nvt["T"])
    T_npt = np.asarray(npt["T"])

    def _plot_U_ah(ax, T, U_mfb, U_md, U_md_err, key_mfb, key_md, band_color):
        ax.plot(T, np.asarray(U_mfb), **style[key_mfb])
        if U_md is not None:
            U_md = np.asarray(U_md)
            ax.plot(T, U_md, **style[key_md])
            if U_md_err is not None:
                err = np.asarray(U_md_err)
                ax.fill_between(
                    T,
                    U_md - err,
                    U_md + err,
                    color=band_color,
                    alpha=0.2,
                    linewidth=0,
                )

    _plot_U_ah(
        ax,
        T_nvt,
        nvt["U_ah_mfb"],
        nvt.get("U_ah_md"),
        nvt.get("U_ah_md_err"),
        "nvt_mfb",
        "nvt_md",
        "k",
    )
    _plot_U_ah(
        ax,
        T_npt,
        npt["U_ah_mfb"],
        npt.get("U_ah_md"),
        npt.get("U_ah_md_err"),
        "npt_mfb",
        "npt_md",
        "k",
    )
    ax.set_ylabel(r"$U_{\mathrm{ah}}$ [meV/atom]", labelpad=1)
    ax.tick_params(
        axis="x", labelbottom=False
    )  # shared T axis: only the bottom row shows it
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    _apply_fontsizes(ax, fontsizes)
    _caption(ax, "Anharmonic energy")

    # (row 2) NVT pressure
    ax = axes[1, 1]
    ax.plot(T_nvt, np.asarray(nvt["P_mfb"]) / 1.0e4, **style["nvt_mfb"])
    if nvt.get("P_md") is not None:
        ax.plot(T_nvt, np.asarray(nvt["P_md"]) / 1.0e4, **style["nvt_md"])
    ax.set_ylabel("P [GPa]", labelpad=1)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(
        axis="x", labelbottom=False
    )  # shared T axis: only the bottom row shows it
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    _apply_fontsizes(ax, fontsizes)
    _caption(ax, "Pressure")

    # (row 3) NPT lattice expansion
    ax = axes[2, 1]
    ax.plot(
        T_npt, (np.asarray(npt["eps_mfb"]) - 1.0) * 100.0, **style["npt_mfb"]
    )
    if npt.get("eps_md") is not None:
        ax.plot(
            T_npt, (np.asarray(npt["eps_md"]) - 1.0) * 100.0, **style["npt_md"]
        )
    ax.set_xlabel(
        "T [K]", labelpad=1
    )  # shared T axis: only this bottom row gets it
    ax.set_ylabel(r"$\Delta a / a_0$ [%]", labelpad=1)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    _apply_fontsizes(ax, fontsizes)
    _caption(ax, "Lattice expansion")

    # shared legend for every thermo panel, in the same top strip as the
    # colorbar: color = ensemble, linestyle/marker = source
    legend_handles = [
        Line2D([0], [0], **{**style["nvt_mfb"], "label": "NVT, MFB"}),
        Line2D([0], [0], **{**style["nvt_md"], "label": "NVT, MD"}),
        Line2D([0], [0], **{**style["npt_mfb"], "label": "NPT, MFB"}),
        Line2D([0], [0], **{**style["npt_md"], "label": "NPT, MD"}),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=2,
        fontsize=fontsizes["legend"],
        frameon=False,
        handlelength=1.4,
        columnspacing=0.9,
        handletextpad=0.4,
        labelspacing=0.25,
        borderaxespad=0.0,
    )

    fig.align_ylabels(axes[:, 1])
    _close_row_gaps(fig, axes)
    _close_column_gap(fig, axes, cbar_ax, legend_ax)
    _trim_outer_whitespace(fig)

    if savepath:
        fig.savefig(savepath, dpi=400, bbox_inches="tight", pad_inches=0)

    return fig, axes
