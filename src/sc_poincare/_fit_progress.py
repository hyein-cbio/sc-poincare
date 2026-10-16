"""AnnData progress snapshots and cell-identity-bound MU checkpoints."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json

import numpy as np

SCHEMA = 'sc-poincare-anndata-mu-v1'


@dataclass(frozen=True)
class PoincareProgress:
    """Independent raw training copies; callbacks never receive display isometries.

    MU ``checkpoint`` wraps an Adam state plus AnnData cell-order provenance.
    RFA has no exact optimizer-resume contract and reports checkpoint=None.
    Exceptions abort the call before its output slots are written.
    """
    method: str
    epoch: int
    loss: float
    best_loss: float
    best_epoch: int
    current: np.ndarray
    best: np.ndarray
    checkpoint: dict | None
    initialization: str
    initialization_reason: str = ''
    initialization_diagnostics: dict = field(default_factory=dict)

    @property
    def coordinates(self):
        return self.current

    @property
    def best_coordinates(self):
        return self.best


def coordinates_sha256(coordinates):
    raw = np.asarray(coordinates)
    if raw.dtype.kind not in 'biuf':
        raise ValueError('Coordinate provenance requires real numeric values.')
    return hashlib.sha256(np.ascontiguousarray(raw, dtype=np.float64).tobytes()).hexdigest()


def annotation_sha256(values):
    array = np.asarray(values)
    if array.dtype.kind in 'biuf':
        payload = b'numeric\0' + np.ascontiguousarray(array, dtype=np.float64).tobytes()
    else:
        payload = b'text\0' + json.dumps([str(x) for x in array], ensure_ascii=False,
                                         separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def _owned_metadata(adata, basis):
    key = basis[2:] if basis.startswith('X_') else basis
    metadata = adata.uns.get(key)
    if not isinstance(metadata, dict):
        return None
    version = metadata.get('schema_version')
    return metadata if isinstance(version, (int, np.integer)) and version == 2 else None


def record_root_display(adata, basis, destination, coordinates, root):
    metadata = _owned_metadata(adata, basis)
    if metadata is None:
        return
    value = None if root is None else (int(root) if isinstance(root, (int,np.integer)) else str(root))
    record = dict(transform='root_isometry', root=value, source_basis=basis,
                  coordinates_sha256=coordinates_sha256(coordinates))
    metadata.setdefault('display_transforms', {})[destination] = record
    if destination == basis:
        metadata.update(coordinate_space='root_isometry', display_root=value,
                        display_coordinates_sha256=record['coordinates_sha256'])


def record_annotation(adata, basis, name, kind, **parameters):
    metadata = _owned_metadata(adata, basis)
    if metadata is not None:
        metadata.setdefault('derived_fields', {})[name] = dict(kind=kind, basis=basis,
            source_coordinates_sha256=coordinates_sha256(adata.obsm[basis]),
            annotation_sha256=annotation_sha256(adata.obs[name]), **parameters)


def observation_provenance(adata):
    names = list(adata.obs_names)
    valid = bool(adata.obs_names.is_unique and all(isinstance(x, str) and x.strip() for x in names))
    encoded = json.dumps([str(x) for x in names], ensure_ascii=False, separators=(',', ':')).encode()
    return dict(n_obs=int(adata.n_obs), obs_names_sha256=hashlib.sha256(encoded).hexdigest(),
                unique_nonempty_obs_names=valid)


def _digest(state, provenance):
    data = json.dumps(provenance, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(data + b'\0' + state['checkpoint_sha256'].encode()).hexdigest()


def wrap_mu_checkpoint(state, provenance):
    provenance = deepcopy(provenance)
    return dict(schema=SCHEMA, state=state, provenance=provenance,
                integrity_sha256=_digest(state, provenance))


def unwrap_mu_checkpoint(checkpoint, adata):
    if (not isinstance(checkpoint, dict) or checkpoint.get('schema') != SCHEMA
            or not isinstance(checkpoint.get('state'), dict)
            or not isinstance(checkpoint.get('provenance'), dict)):
        raise ValueError('MU AnnData resume requires a checkpoint from progress_callback; '
                         'use experimental.fit_graph_stress for bare low-level states.')
    state, provenance = checkpoint['state'], checkpoint['provenance']
    try:
        intact = checkpoint.get('integrity_sha256') == _digest(state, provenance)
    except (KeyError, TypeError, ValueError, AttributeError):
        intact = False
    if not intact:
        raise ValueError('AnnData checkpoint provenance integrity mismatch.')
    observed = observation_provenance(adata)
    if not observed['unique_nonempty_obs_names'] or not provenance.get('unique_nonempty_obs_names'):
        raise ValueError('Verified MU resume requires unique, nonempty obs_names in the original and resumed input.')
    for key in ('n_obs', 'obs_names_sha256'):
        if observed[key] != provenance.get(key):
            raise ValueError('Checkpoint cell identities/order differ; align the original cells explicitly.')
    return state, provenance
