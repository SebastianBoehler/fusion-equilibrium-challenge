# Organizer dependency provenance

Source: https://github.com/Sophelio/fusion-equilibrium-challenge-starter

Revision: `5b467988d2e5f28e20be490a926d0a71649de28d`

The files under `vendor/fusion_starter/` are copied unchanged from this revision,
including the scorer masks and timestamp correction. The upstream MIT license is
in `vendor/fusion_starter/LICENSE`; our own code is MIT-licensed in the root `LICENSE`. No demo shots or MAST equilibrium labels are included. The organizer scorer ships
`fusion_scoring/masks/mast_envelope.npz` (aggregated from label contours); only the
unmodified scorer uses it, never our MAST prediction path.

The complete initial starter checkout is retained locally in ignored `.journal/`.
The `upstream` remote retains the organizer history; our `main` has an independent
participant history.
