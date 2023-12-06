"""Analysis tasks module."""

from .annotate_bad_channels_and_segments import (  # noqa: F401
    annotate_bad_channels_and_segments,
    bridges_and_autobads,
    view_annotated_raw,
)
from .convert_xdf_to_fiff import convert_xdf_to_fiff  # noqa: F401
from .epochs_evoked import create_epochs_evoked_and_behavioral_metadata  # noqa: F401
from .ica_decomposition import (  # noqa: F401
    apply_ica,
    compare_labels,
    fit_icas,
    label_components,
)
from .create_behavioral_metadata import create_behavioral_metadata # noqa: F401
