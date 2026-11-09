"""Centre slices and line cuts of a bond density, optionally against an MD
density on the same grid."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

__all__ = [
    "DEFAULT_FONTSIZES",
    "compute_equilibrium_bond_length",
    "plot_bond_density_slices",
]

DEFAULT_FONTSIZES = {
    "suptitle": 16,
    "title": 13,
    "label": 13,
    "tick": 10,
    "legend": 11,
    "clabel": 11,
    "caption": 11,
}


def compute_equilibrium_bond_length(bond_grid, bond_density):
    """<|b|> of a density on a bond grid of shape (nx, ny, nz, 3)."""
    return float(np.sum(bond_density * np.linalg.norm(bond_grid, axis=-1)))


def plot_bond_density_slices(
    disps,
    rho_mfbm,
    rho_md=None,
    temperature=None,
    label=None,
    n_levels=8,
    planes=("xy", "xz", "yz"),
    fontsizes=None,
):
    """
    One column per plane: the centre slice as contours (MD filled with the
    model as lines, if rho_md is given) and, below it, the cut through the
    centre of that slice.

    Parameters
    ----------
    disps : np.ndarray
        Displacement axis shared by x, y and z.
    rho_mfbm, rho_md : np.ndarray, shape (len(disps),) * 3
    label : str, optional
        Added to the title, e.g. "Al-Fe, shell 1".

    Returns
    -------
    fig, axes
        axes has shape (2, len(planes)).
    """
    fs = {**DEFAULT_FONTSIZES, **(fontsizes or {})}
    c = len(disps) // 2
    take = {
        "xy": lambda r: r[:, :, c],
        "xz": lambda r: r[:, c, :],
        "yz": lambda r: r[c, :, :],
    }
    names = {
        "xy": ("x", "y", "z"),
        "xz": ("x", "z", "y"),
        "yz": ("y", "z", "x"),
    }

    slices = {}
    for plane in planes:
        model = take[plane](rho_mfbm)
        md = take[plane](rho_md) if rho_md is not None else None
        slices[plane] = (
            model / model.sum(),
            md / md.sum() if md is not None else None,
        )
    filled = [md if md is not None else model for model, md in slices.values()]
    levels = np.linspace(
        min(f.min() for f in filled),
        max(f.max() for f in filled),
        n_levels + 1,
    )

    fig, axes = plt.subplots(
        2,
        len(planes),
        figsize=(6 * len(planes), 9),
        gridspec_kw={"height_ratios": [3, 1]},
        squeeze=False,
    )
    H, V = np.meshgrid(disps, disps, indexing="ij")
    for col, plane in enumerate(planes):
        model, md = slices[plane]
        h, v, fixed = names[plane]
        top = axes[0, col]
        bottom = axes[1, col]
        if md is not None:
            mappable = top.contourf(H, V, md, levels=levels, cmap="coolwarm")
            lines = top.contour(
                H, V, model, levels=levels, cmap="inferno", linewidths=1.5
            )
            top.clabel(lines, inline=True, fontsize=fs["clabel"], fmt="%.1e")
        else:
            mappable = top.contourf(H, V, model, levels=levels, cmap="viridis")
        top.set(
            title=f"{fixed}_disp = {disps[c]:.2f} A ({plane} slice)",
            xlabel=f"{h}_disp (A)",
            ylabel=f"{v}_disp (A)",
        )
        bottom.plot(disps, model[:, c], color="tab:red", label="MFBM")
        if md is not None:
            bottom.plot(disps, md[:, c], color="tab:blue", label="MD")
        bottom.set(
            title=f"cut at {v}_disp = {disps[c]:.2f} A",
            xlabel=f"{h}_disp (A)",
            ylabel="density",
        )
        bottom.legend(fontsize=fs["legend"])
        bottom.yaxis.set_major_formatter(FormatStrFormatter("%.0e"))
        for ax in (top, bottom):
            ax.title.set_fontsize(fs["title"])
            ax.xaxis.label.set_fontsize(fs["label"])
            ax.yaxis.label.set_fontsize(fs["label"])
            ax.tick_params(labelsize=fs["tick"])
    ylim = (
        min(ax.get_ylim()[0] for ax in axes[1]),
        max(ax.get_ylim()[1] for ax in axes[1]),
    )
    for ax in axes[1]:
        ax.set_ylim(ylim)

    fig.suptitle(
        "; ".join(
            x
            for x in (f"T = {temperature} K" if temperature else None, label)
            if x
        ),
        fontsize=fs["suptitle"],
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    width = axes[0, 0].get_position().width
    bar = fig.colorbar(
        mappable,
        cax=fig.add_axes(((1 - width) / 2, 0.90, width, 0.03)),
        orientation="horizontal",
        ticks=levels[::2],
        format="%.0e",
    )
    bar.set_label(
        "density" if rho_md is None else "density (MD filled, MFBM lines)",
        fontsize=fs["label"],
    )
    bar.ax.tick_params(labelsize=fs["tick"])
    return fig, axes
