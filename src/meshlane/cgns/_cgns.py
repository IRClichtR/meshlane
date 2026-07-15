"""
CGNS <https://cgns.github.io/>

The reader follows the CGNS/HDF5 (a.k.a. ADF-on-HDF5) layout: every CGNS node
is an HDF5 group carrying ``label``/``name``/``type`` attributes (optionally
prefixed with a leading space) and storing its payload in a ``" data"`` dataset
(note the leading space). We walk that tree by node label rather than by fixed
group names, which keeps the reader robust to the many ways CGNS files name
their bases, zones and element sections.

References:
- CGNS standard: https://cgns.github.io/standard/SIDS/
- CGNS/HDF5 file mapping: https://cgns.github.io/standard/MLL/CGNS_MLL.html
"""

import numpy as np

from .._common import warn
from .._exceptions import ReadError
from .._helpers import register_format
from .._mesh import Mesh

# CGNS ElementType_t enum value -> (meshio cell type, nodes per cell).
# A nodes-per-cell of None marks the two variable-length "poly" element types,
# NGON_n (polygonal faces) and NFACE_n (polyhedral cells), which are handled
# separately. Linear element node orderings match meshio's directly; higher-
# order orderings are assumed to coincide with CGNS' and are not permuted.
_cgns_to_meshio_type = {
    2: ("vertex", 1),  # NODE
    3: ("line", 2),  # BAR_2
    4: ("line3", 3),  # BAR_3
    5: ("triangle", 3),  # TRI_3
    6: ("triangle6", 6),  # TRI_6
    7: ("quad", 4),  # QUAD_4
    8: ("quad8", 8),  # QUAD_8
    9: ("quad9", 9),  # QUAD_9
    10: ("tetra", 4),  # TETRA_4
    11: ("tetra10", 10),  # TETRA_10
    12: ("pyramid", 5),  # PYRA_5
    13: ("pyramid14", 14),  # PYRA_14
    14: ("wedge", 6),  # PENTA_6
    15: ("wedge15", 15),  # PENTA_15
    16: ("wedge18", 18),  # PENTA_18
    17: ("hexahedron", 8),  # HEXA_8
    18: ("hexahedron20", 20),  # HEXA_20
    19: ("hexahedron27", 27),  # HEXA_27
    21: ("pyramid13", 13),  # PYRA_13
    22: ("polygon", None),  # NGON_n
    23: ("polyhedron", None),  # NFACE_n
}

NGON_N = 22
NFACE_N = 23


def _read_attr(group, name):
    """Return an attribute that may carry a leading space (CGNS/ADF convention)."""
    for key in (f" {name}", name):
        if key in group.attrs:
            return group.attrs[key]
    return None


def _node_label(group):
    """Return the CGNS label (node type, e.g. ``"Zone_t"``) of an HDF5 group."""
    value = _read_attr(group, "label")
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("ascii", "replace")
    return str(value).strip().strip("\x00")


def _find_first_child_with_label(group, label):
    import h5py

    for name in group:
        child = group[name]
        if isinstance(child, h5py.Group) and _node_label(child) == label:
            return child
    raise ReadError(f'CGNS: no child with label "{label}" under "{group.name}".')


def _children_with_label(group, label):
    import h5py

    return [
        group[name]
        for name in group
        if isinstance(group[name], h5py.Group) and _node_label(group[name]) == label
    ]


def _node_data(group):
    """Return the payload stored in a node's ``" data"`` dataset."""
    return np.asarray(group[" data"][()])


def _index_array(node):
    """Read a node's ``" data"`` as a flat array of 1-based CGNS indices."""
    return _node_data(node).ravel().astype(np.int64)


def _read_string_node(group):
    """Decode a ``C1`` node whose ``" data"`` holds a null-terminated char array."""
    raw = _node_data(group).ravel()
    text = bytes(int(b) & 0xFF for b in raw)
    return text.split(b"\x00", 1)[0].decode("ascii", "replace").strip()


def _read_coordinates(zone, phys_dim):
    grid = _find_first_child_with_label(zone, "GridCoordinates_t")
    names = ["CoordinateX", "CoordinateY", "CoordinateZ"][:phys_dim]
    columns = [np.asarray(grid[name][" data"][()], dtype=np.float64) for name in names]
    return np.column_stack(columns)


def _resolve_polyhedra(cell_offsets, cell_faces, face_offsets, face_nodes):
    """Build polyhedron cell blocks from NFACE_n cells and NGON_n faces.

    Each NFACE_n cell lists signed 1-based face indices; the sign encodes face
    orientation and is dropped. Each referenced NGON_n face lists 1-based node
    indices. The result follows meshio's polyhedron layout: a list (per cell)
    of lists (per face) of 0-based node-index arrays, grouped into
    ``polyhedron{n}`` blocks by the cell's unique node count (as in the vtu
    reader).
    """
    blocks = {}
    n_cells = len(cell_offsets) - 1
    for i in range(n_cells):
        faces = []
        for face_ref in cell_faces[cell_offsets[i] : cell_offsets[i + 1]]:
            face_idx = abs(int(face_ref)) - 1
            nodes = face_nodes[face_offsets[face_idx] : face_offsets[face_idx + 1]] - 1
            faces.append(nodes)
        n_unique = np.unique(np.concatenate(faces)).size
        blocks.setdefault(f"polyhedron{n_unique}", []).append(faces)
    return list(blocks.items())


def _read_elements(zone):
    cells = []

    # NGON_n faces must be read before NFACE_n cells can be resolved, so keep
    # their raw connectivity/offsets around for a second pass.
    ngon_offsets = None
    ngon_conn = None
    nface_offsets = None
    nface_conn = None

    for section in _children_with_label(zone, "Elements_t"):
        code = int(_node_data(section).ravel()[0])
        info = _cgns_to_meshio_type.get(code)
        if info is None:
            warn(
                f"CGNS: unsupported element type {code}; "
                f'section "{section.name}" skipped.'
            )
            continue

        meshio_type, nodes_per_cell = info

        if code == NGON_N:
            conn = _index_array(section["ElementConnectivity"])
            offsets = _index_array(section["ElementStartOffset"])
            polygons = [
                conn[offsets[i] : offsets[i + 1]] - 1 for i in range(len(offsets) - 1)
            ]
            cells.append(("polygon", polygons))
            ngon_offsets, ngon_conn = offsets, conn
        elif code == NFACE_N:
            nface_conn = _index_array(section["ElementConnectivity"])
            nface_offsets = _index_array(section["ElementStartOffset"])
        else:
            start, end = _index_array(section["ElementRange"])
            n_cells = int(end - start + 1)
            conn = _index_array(section["ElementConnectivity"])
            cells.append((meshio_type, conn.reshape(n_cells, nodes_per_cell) - 1))

    if nface_offsets is not None:
        if ngon_offsets is None:
            raise ReadError(
                "CGNS: NFACE_n section found without an NGON_n section to resolve it."
            )
        cells.extend(
            _resolve_polyhedra(nface_offsets, nface_conn, ngon_offsets, ngon_conn)
        )

    return cells


def read(filename):
    """Read an unstructured CGNS mesh from a CGNS/HDF5 file.

    Limitations:
    - Only the CGNS/HDF5 (ADF-on-HDF5) file layout is supported, where every
      CGNS node is an HDF5 group and its payload lives in a ``" data"`` dataset.
    - Only ``Unstructured`` zones are read; structured zones are rejected.
    - Exactly one base and one zone are read: the first ``CGNSBase_t`` node and,
      within it, the first ``Zone_t`` node. Additional bases or zones are
      ignored.
    """
    import h5py

    with h5py.File(filename, "r") as f:
        base = _find_first_child_with_label(f, "CGNSBase_t")
        dims = _node_data(base).ravel()
        if dims.size < 2:
            raise ReadError("CGNS: CGNSBase_t data must have at least 2 entries.")
        phys_dim = int(dims[1])

        zone = _find_first_child_with_label(base, "Zone_t")

        zone_type = _read_string_node(_find_first_child_with_label(zone, "ZoneType_t"))
        if zone_type != "Unstructured":
            raise ReadError(
                f'CGNS: unsupported zone type "{zone_type}"; '
                "only Unstructured zones are supported."
            )

        points = _read_coordinates(zone, phys_dim)
        cells = _read_elements(zone)

    return Mesh(points, cells)


def write(filename, mesh, compression="gzip", compression_opts=4):
    import h5py

    f = h5py.File(filename, "w")

    base = f.create_group("Base")

    # TODO something is missing here

    zone1 = base.create_group("Zone1")
    coords = zone1.create_group("GridCoordinates")

    # write points
    coord_x = coords.create_group("CoordinateX")
    coord_x.create_dataset(
        " data",
        data=mesh.points[:, 0],
        compression=compression,
        compression_opts=compression_opts,
    )
    coord_y = coords.create_group("CoordinateY")
    coord_y.create_dataset(
        " data",
        data=mesh.points[:, 1],
        compression=compression,
        compression_opts=compression_opts,
    )
    coord_z = coords.create_group("CoordinateZ")
    coord_z.create_dataset(
        " data",
        data=mesh.points[:, 2],
        compression=compression,
        compression_opts=compression_opts,
    )

    # write cells
    # TODO write cells other than tetra
    elems = zone1.create_group("GridElements")
    rnge = elems.create_group("ElementRange")
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            rnge.create_dataset(
                " data",
                data=[1, cell_block.data.shape[0]],
                compression=compression,
                compression_opts=compression_opts,
            )
    conn = elems.create_group("ElementConnectivity")
    for cell_block in mesh.cells:
        if cell_block.type == "tetra":
            conn.create_dataset(
                " data",
                data=cell_block.data.reshape(-1) + 1,
                compression=compression,
                compression_opts=compression_opts,
            )


register_format("cgns", [".cgns"], read, {"cgns": write})
