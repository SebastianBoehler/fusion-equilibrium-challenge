# MAST machine geometry (public)

`mast_pf_passive.json`: the EFIT passive-structure geometry of MAST. Each element gives
centre R and Z, width and height, in metres.

- **Source:** FAIR-MAST level 2, `pf_passive` group
  (`s3.echo.stfc.ac.uk/mast/level2/shots/30420.zarr/pf_passive`). Its source attribute is
  `mast-geometry/sources/pfcoils.dat_M4`, valid for shots 0–40000.
- **What was read:** geometry arrays only, from a public machine description. No currents,
  magnetics or equilibrium data were read.
- **Known data quirk:** the coil-case widths and heights are signed and partly swapped, so
  absolute values are used.
- **Not in the file:** resistivities. They come from the literature: Inconel-600
  1.03e-6 Ω·m, stainless steel 7.4e-7 Ω·m (Berkery et al., PPCF 63 055014, 2021; the
  FreeGSNKE MAST-U machine files).
