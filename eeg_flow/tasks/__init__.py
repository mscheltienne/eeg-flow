"""Analysis tasks module."""

from .annotate_bad_channels_and_segments import (  # noqa: F401
    annotate_bad_channels_and_segments,
    bridges_and_autobads,
    view_annotated_raw,
)
from .convert_xdf_to_fiff import convert_xdf_to_fiff  # noqa: F401
from .create_behavioral_metadata import create_behavioral_metadata  # noqa: F401
from .epochs_evoked import (  # noqa: F401
    create_epochs_evoked_and_behavioral_metadata,
    response_to_CSD,
    stimlocked_to_CSD,
)
from .epochs_evoked_for_freqresponse import (  # noqa: F401
    create_epochs_evoked_and_behavioral_metadata_response,
)
from .ica_decomposition import (  # noqa: F401
    apply_ica,
    apply_ica_interpolate,
    apply_ica_reref_EOG,
    compare_labels,
    fit_icas,
    label_components,
)
