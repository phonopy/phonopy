# SPDX-License-Identifier: BSD-3-Clause
"""Figure-level plotting of phonon calculation results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from phonopy.phonon.dos import get_dos_frequency_range

if TYPE_CHECKING:
    from phonopy.phonon.band_structure import BandStructure
    from phonopy.phonon.dos import ProjectedDos, TotalDos
    from phonopy.phonon.thermal_displacement import ThermalDisplacements
    from phonopy.phonon.thermal_properties import ThermalProperties


def _set_ax_ticks_both(ax: Any) -> None:
    """Draw ticks on all four sides, pointing inward."""
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_tick_params(which="both", direction="in")
    ax.yaxis.set_tick_params(which="both", direction="in")


def _band_image_grid(fig: Any, n: int) -> Any:
    """Return ImageGrid axes for n horizontally aligned panels."""
    from mpl_toolkits.axes_grid1 import ImageGrid

    return ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(1, n),
        axes_pad=0.11,
        label_mode="L",
    )


def plot_band_structure(band_structure: BandStructure) -> Any:
    """Plot band structure into a new matplotlib figure.

    Parameters
    ----------
    band_structure : BandStructure
        Band structure result to plot.

    Returns
    -------
    matplotlib.pyplot
        The ``matplotlib.pyplot`` module. Call ``.show()`` on it to
        display the figure.

    """
    import matplotlib.pyplot as plt

    if band_structure.is_legacy_plot:
        fig, axs = plt.subplots(1, 1)
    else:
        n = len([x for x in band_structure.path_connections if not x])
        fig = plt.figure()
        axs = _band_image_grid(fig, n)
    band_structure.plot(axs)
    return plt


def plot_band_structure_and_dos(
    band_structure: BandStructure,
    total_dos: TotalDos | None = None,
    projected_dos: ProjectedDos | None = None,
    pdos_indices: Sequence[Sequence[int]] | None = None,
) -> Any:
    """Plot band structure and DOS side by side into a new figure.

    Parameters
    ----------
    band_structure : BandStructure
        Band structure result to plot.
    total_dos : TotalDos, optional
        Total DOS result. Used and required when ``pdos_indices`` is
        None.
    projected_dos : ProjectedDos, optional
        Projected DOS result. Used and required when ``pdos_indices``
        is given.
    pdos_indices : list of list, optional
        Sets of indices of atoms whose projected DOS are summed over.
        The indices start with 0. An example is
        ``pdos_indices=[[0, 1], [2, 3, 4, 5]]``. Default is None,
        which plots the total DOS.

    Returns
    -------
    matplotlib.pyplot
        The ``matplotlib.pyplot`` module. Call ``.show()`` on it to
        display the figure.

    """
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    if pdos_indices is None and total_dos is None:
        raise RuntimeError("total_dos has to be given.")
    if pdos_indices is not None and projected_dos is None:
        raise RuntimeError("projected_dos has to be given.")

    if band_structure.is_legacy_plot:
        import matplotlib.gridspec as gridspec

        # plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax2 = plt.subplot(gs[0, 1])
        if pdos_indices is None:
            assert total_dos is not None
            total_dos.plot(ax2, ylabel="", draw_grid=False, flip_xy=True)
        else:
            assert projected_dos is not None
            projected_dos.plot(
                ax2, indices=pdos_indices, ylabel="", draw_grid=False, flip_xy=True
            )
        ax2.set_xlim(left=0, right=None)
        plt.setp(ax2.get_yticklabels(), visible=False)

        ax1 = plt.subplot(gs[0, 0], sharey=ax2)
        band_structure.plot(ax1)

        plt.subplots_adjust(wspace=0.03)
        plt.tight_layout()
    else:
        n = len([x for x in band_structure.path_connections if not x]) + 1
        fig = plt.figure()
        axs = _band_image_grid(fig, n)
        band_structure.plot(axs[:-1])

        if pdos_indices is None:
            assert total_dos is not None
            total_dos.plot(axs[-1], xlabel="", ylabel="", draw_grid=False, flip_xy=True)
        else:
            assert projected_dos is not None
            projected_dos.plot(
                axs[-1],
                indices=pdos_indices,
                xlabel="",
                ylabel="",
                draw_grid=False,
                flip_xy=True,
            )
        last_axs = cast(Axes, axs[-1])
        xlim = last_axs.get_xlim()
        ylim = last_axs.get_ylim()
        aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0]) * 3
        last_axs.set_aspect(aspect)
        last_axs.axhline(y=0, linestyle=":", linewidth=0.5, color="b")
        last_axs.set_xlim(left=0, right=None)

    return plt


def plot_total_dos(
    total_dos: TotalDos,
    xlabel: str | None = None,
    ylabel: str | None = None,
    with_tight_frequency_range: bool = False,
) -> Any:
    """Plot total DOS into a new matplotlib figure.

    Parameters
    ----------
    total_dos : TotalDos
        Total DOS result to plot.
    xlabel : str, optional
        x-label of the plot. Default is None, which puts a default
        x-label.
    ylabel : str, optional
        y-label of the plot. Default is None, which puts a default
        y-label.
    with_tight_frequency_range : bool, optional
        Plot with a tight frequency range. Default is False.

    Returns
    -------
    matplotlib.pyplot
        The ``matplotlib.pyplot`` module. Call ``.show()`` on it to
        display the figure.

    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    total_dos.plot(ax, xlabel=xlabel, ylabel=ylabel, draw_grid=False)
    if with_tight_frequency_range:
        assert total_dos.dos is not None
        fmin, fmax = get_dos_frequency_range(total_dos.frequency_points, total_dos.dos)
        ax.set_xlim(left=fmin, right=fmax)
    ax.set_ylim(bottom=0, top=None)

    return plt


def plot_projected_dos(
    projected_dos: ProjectedDos,
    pdos_indices: Sequence[Sequence[int]] | None = None,
    legend: Sequence[str] | None = None,
    legend_prop: dict | None = None,
    legend_frameon: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    with_tight_frequency_range: bool = False,
) -> Any:
    """Plot projected DOS into a new matplotlib figure.

    Parameters
    ----------
    projected_dos : ProjectedDos
        Projected DOS result to plot.
    pdos_indices : list of list, optional
        Sets of indices of atoms whose projected DOS are summed over.
        The indices start with 0. An example is
        ``pdos_indices=[[0, 1], [2, 3, 4, 5]]``. Default is None,
        which means ``pdos_indices=[[i] for i in range(natom)]``.
    legend : list of instances such as str or int, optional
        The str(instance) are shown in legend.
        It has to be len(pdos_indices)==len(legend). Default is None.
        When None, legend is not shown.
    legend_prop : dict, optional
        Legend properties of matplotlib. Default is None.
    legend_frameon : bool, optional
        Legend with frame or not. Default is True.
    xlabel : str, optional
        x-label of plot. Default is None, which puts a default x-label.
    ylabel : str, optional
        y-label of plot. Default is None, which puts a default y-label.
    with_tight_frequency_range : bool, optional
        Plot with tight frequency range. Default is False.

    Returns
    -------
    matplotlib.pyplot
        The ``matplotlib.pyplot`` module. Call ``.show()`` on it to
        display the figure.

    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    projected_dos.plot(
        ax,
        indices=pdos_indices,
        legend=legend,
        legend_prop=legend_prop,
        legend_frameon=legend_frameon,
        xlabel=xlabel,
        ylabel=ylabel,
        draw_grid=False,
    )

    if with_tight_frequency_range:
        assert projected_dos.projected_dos is not None
        fmin, fmax = get_dos_frequency_range(
            projected_dos.frequency_points, projected_dos.projected_dos.sum(axis=0)
        )
        ax.set_xlim(left=fmin, right=fmax)
    ax.set_ylim(bottom=0, top=None)

    return plt


def plot_thermal_properties(
    thermal_properties: ThermalProperties,
    xlabel: str | None = None,
    ylabel: str | None = None,
    with_grid: bool = True,
    divide_by_Z: bool = False,
    legend_style: str | None = "normal",
) -> Any:
    """Plot thermal properties into a new matplotlib figure.

    Parameters
    ----------
    thermal_properties : ThermalProperties
        Thermal properties result to plot.
    xlabel : str, optional
        Label used for x-axis.
    ylabel : str, optional
        Label used for y-axis.
    with_grid : bool, optional
        With grid or not. Default is True.
    divide_by_Z : bool, optional
        Divide thermal properties by number of formula units of primitive
        cell. Default is False.
    legend_style : str, optional
        "normal", "compact", None. None will not show legend.

    Returns
    -------
    matplotlib.pyplot
        The ``matplotlib.pyplot`` module. Call ``.show()`` on it to
        display the figure.

    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _set_ax_ticks_both(ax)

    thermal_properties.plot(
        ax,
        xlabel=xlabel,
        ylabel=ylabel,
        with_grid=with_grid,
        divide_by_Z=divide_by_Z,
        legend_style=legend_style,
    )

    assert thermal_properties.temperatures is not None
    temps = thermal_properties.temperatures
    ax.set_xlim(left=0, right=temps[-1])

    return plt


def plot_thermal_displacements(
    thermal_displacements: ThermalDisplacements,
    is_legend: bool = False,
) -> Any:
    """Plot thermal displacements into a new matplotlib figure.

    Parameters
    ----------
    thermal_displacements : ThermalDisplacements
        Thermal displacements result to plot.
    is_legend : bool, optional
        Show legend when True. Default is False.

    Returns
    -------
    matplotlib.pyplot
        The ``matplotlib.pyplot`` module. Call ``.show()`` on it to
        display the figure.

    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _set_ax_ticks_both(ax)

    thermal_displacements.plot(ax, is_legend=is_legend)

    assert thermal_displacements.temperatures is not None
    temps = thermal_displacements.temperatures
    ax.set_xlim(left=0, right=temps[-1])

    return plt
