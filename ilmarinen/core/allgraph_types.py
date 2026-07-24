"""Small shared value types for the AllGraph controller and its mixins."""
from collections import namedtuple

# Sweep-invariant context for training one candidate contract on a held-out split. Bundles the eight args
# that _train_candidate_contract[_impl] need so the width/depth/seed sweep (which varies self.width/depth/
# seed, NOT these) passes a single object instead of an eight-argument list at every score call.
_SweepCtx = namedtuple("_SweepCtx", "data contract tr va task n_out epochs edge_cutoff")

# When building graph edges from positions (no explicit edges given), the cutoff is SCALE-ADAPTIVE by
# default: this many times the median nearest-neighbor distance, so edge density is invariant to the
# coordinate scale (molecular Angstrom, unit cube, ...). Pass an explicit edge_cutoff for an absolute
# distance (e.g. 3.0 for molecular Angstrom data).
_EDGE_NN_FACTOR = 1.5
