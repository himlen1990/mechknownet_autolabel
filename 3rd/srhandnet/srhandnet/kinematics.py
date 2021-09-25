mpii_keypoint_names = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "little_mcp",
    "little_pip",
    "little_dip",
    "little_tip",
)


abbreviated_mpii_keypoint_names = (
    'w',  # 0
    't0', 't1', 't2', 't3',  # 4
    'i0', 'i1', 'i2', 'i3',  # 8
    'm0', 'm1', 'm2', 'm3',  # 12
    'r0', 'r1', 'r2', 'r3',  # 16
    'l0', 'l1', 'l2', 'l3',  # 20
)


_index_to_mpii_keypoint_name = {
    i: name for i, name in enumerate(mpii_keypoint_names)
}


_mpii_keypoint_name_to_index = {
    name: i for i, name in enumerate(mpii_keypoint_names)
}


_abbr_to_mpii_keypoint_name = {
    abbr: name for abbr, name in zip(
        abbreviated_mpii_keypoint_names, mpii_keypoint_names)
}


def index_to_mpii_keypoint_name(index):
    return _index_to_mpii_keypoint_name[index]


def indices_to_mpii_keypoint_names(indices):
    return [_index_to_mpii_keypoint_name[i] for i in indices]


def normalize_mpii_keypoint_names(values):
    names = []
    for value in values:
        if isinstance(value, int):
            value = _index_to_mpii_keypoint_name[value]
        elif isinstance(value, str):
            value = _abbr_to_mpii_keypoint_name[value]
        else:
            raise ValueError('Unsupported type {}'.format(type(value)))
        names.append(value)
    return names


def normalize_mpii_keypoint_names_to_indices(values):
    indices = []
    for value in values:
        if isinstance(value, int):
            pass
        elif isinstance(value, str):
            value = value.lower()
            if value in _abbr_to_mpii_keypoint_name:
                value = _mpii_keypoint_name_to_index[
                    _abbr_to_mpii_keypoint_name[value]]
            elif value in _mpii_keypoint_name_to_index:
                value = _mpii_keypoint_name_to_index[value]
            else:
                raise ValueError('Unsupported finger joint name: {}'
                                 .format(value))
        else:
            raise ValueError('Unsupported type: {}'.format(type(value)))
        indices.append(value)
    return indices
