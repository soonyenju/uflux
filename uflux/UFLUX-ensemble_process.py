# Global maps

fig, axes = setup_canvas(5, 1, figsize = (8, 15))

ax = axes[0]
gpp.plot(ax = ax, cmap = 'YlGn', vmin = 0, vmax = 10, cbar_kwargs={'shrink': 0.8, 'label': 'GPP ($gC \ m^{-2} \ d^{-1}$)'})
world_one.plot(ax = ax, color = 'none', edgecolor = 'k')

ax = axes[1]
reco.plot(ax = ax, cmap = 'YlOrBr', vmin = 0, vmax = 10, cbar_kwargs={'shrink': 0.8, 'label': 'Reco ($gC \ m^{-2} \ d^{-1}$)'})
world_one.plot(ax = ax, color = 'none', edgecolor = 'k')

ax = axes[2]
nee.plot(ax = ax, cmap = 'PRGn_r', vmin = -5, vmax = 5, cbar_kwargs={'shrink': 0.8, 'label': 'NEE ($gC \ m^{-2} \ d^{-1}$)'})
world_one.plot(ax = ax, color = 'none', edgecolor = 'k')

ax = axes[3]
nca['H'].mean(dim = 'prod').plot(ax = ax, cmap = 'hot_r', vmin = 0, vmax = 100, cbar_kwargs={'shrink': 0.8, 'label': 'H ($W \ m^{-2}$)'}) # inferno
world_one.plot(ax = ax, color = 'none', edgecolor = 'k')

ax = axes[4]
nca['LE'].mean(dim = 'prod').plot(ax = ax, cmap = 'Blues', vmin = 0, vmax = 100, cbar_kwargs={'shrink': 0.8, 'label': 'LE ($W \ m^{-2}$)'})
world_one.plot(ax = ax, color = 'none', edgecolor = 'k')

# ax.text(
#     0.05, 0.1,
#     f'GPP', transform = ax.transAxes,
#     fontsize = 14,
#     bbox=dict(
#         facecolor='white',  # background color
#         edgecolor='none',   # or 'black' if you want a border
#         boxstyle='round,pad=0.3',  # rounded corners
#         alpha=1            # optional transparency
#     )
# )

google.download_file(fig, 'UFLUX-maps.jpg', dpi = 600)
# google.download_file(fig, 'UFLUX-maps.pdf')