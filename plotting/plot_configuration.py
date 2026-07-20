import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colorbar import Colorbar

def plot_configuration(axis_object: plt.Axes, legend_location: str = None, legend_size: float = 12.5, 
                       font_size: float = 12, colorbar_object: Colorbar = None) -> plt.Axes:
    """Configuration for most plots used troughout the thesis.
    Encapsulates the legend and the plot itself using black lines.

    Parameters
    ----------
    axis_object
        Original Matplotlib axis object.
    legend_location, optional
        Location of the legend in the plot (also handles if legend is present), by default None.
    legend_size, optional
        Size of the legend, by default 12.5.
    font_size, optional
        Size of the used font, by default 12.
    colorbar_object, optional
        Colorbar object appearing when cmaps are used, by default None.


    Returns
    -------
        Matplotlib axis object with applied configurations.
    """
    for spine in axis_object.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    axis_object.tick_params(axis="both", which="major", length=4, color="black")

    if legend_location:
        leg = axis_object.legend(
            frameon=True,
            fancybox=False,
            facecolor="white",
            loc=legend_location,
            prop={"size": legend_size}
        )

        frame = leg.get_frame()
        frame.set_edgecolor("black")
        frame.set_linewidth(1.2)
        frame.set_alpha(1)

    pref_font = "TeX Gyre Pagella"
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    if pref_font in available_fonts:
        font_name = pref_font
    else:
        font_name = "serif"
        
    axis_object.title.set_fontname(font_name)
    axis_object.xaxis.label.set_fontname(font_name)
    axis_object.yaxis.label.set_fontname(font_name)
    axis_object.title.set_fontsize(font_size)
    axis_object.xaxis.label.set_fontsize(font_size)
    axis_object.yaxis.label.set_fontsize(font_size)

    if colorbar_object:
        colorbar_object.ax.yaxis.label.set_fontname(font_name)
        colorbar_object.ax.yaxis.label.set_fontsize(font_size)

    return axis_object