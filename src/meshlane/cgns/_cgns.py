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

import sys

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

# Inverse of ``_cgns_to_meshio_type`` for the writer, restricted to the
# fixed-size element types. The two variable-length "poly" types (NGON_n/
# NFACE_n) are written through the dedicated polygon/polyhedron path instead.
_meshio_to_cgns_type = {
    meshio_type: (code, nodes_per_cell)
    for code, (meshio_type, nodes_per_cell) in _cgns_to_meshio_type.items()
    if nodes_per_cell is not None
}


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


def _resolve_polyhedra(cell_offsets, cell_faces, face_offsets, face_nodes, face_start):
    """Build polyhedron cell blocks from NFACE_n cells and NGON_n faces.

    Each NFACE_n cell lists signed face references; the sign encodes face
    orientation and is dropped, and the magnitude is the face's global CGNS
    element number. ``face_start`` is the NGON_n section's first element number,
    so ``abs(ref) - face_start`` is the face's 0-based position in that section.
    Each referenced NGON_n face lists 1-based node indices. The result follows
    meshio's polyhedron layout: a list (per cell) of lists (per face) of 0-based
    node-index arrays, grouped into ``polyhedron{n}`` blocks by the cell's unique
    node count (as in the vtu reader).
    """
    blocks = {}
    n_cells = len(cell_offsets) - 1
    for i in range(n_cells):
        faces = []
        for face_ref in cell_faces[cell_offsets[i] : cell_offsets[i + 1]]:
            face_idx = abs(int(face_ref)) - face_start
            nodes = face_nodes[face_offsets[face_idx] : face_offsets[face_idx + 1]] - 1
            faces.append(nodes)
        n_unique = np.unique(np.concatenate(faces)).size
        blocks.setdefault(f"polyhedron{n_unique}", []).append(faces)
    return list(blocks.items())


def _read_elements(zone):
    cells = []

    # NGON_n faces must be read before NFACE_n cells can be resolved, so keep
    # their raw connectivity/offsets (and section start) around for a second pass.
    ngon_offsets = None
    ngon_conn = None
    ngon_start = None
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
            ngon_start = int(_index_array(section["ElementRange"])[0])
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
            _resolve_polyhedra(
                nface_offsets, nface_conn, ngon_offsets, ngon_conn, ngon_start
            )
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


# CGNS/HDF5 (ADF) data-type strings, stored in each node's ``type`` attribute,
# and the numpy dtype each one maps to. ``MT`` marks a node that carries no data.
_ADF_DTYPE = {
    "I4": np.int32,
    "I8": np.int64,
    "R4": np.float32,
    "R8": np.float64,
    "C1": np.int8,
}


def _string_dtype(length):
    """A fixed-length ASCII string dtype, as CGNS uses for name/label/type attrs."""
    import h5py

    return h5py.string_dtype(encoding="ascii", length=length)


def _set_node_attrs(node, name, label, adf_type, with_flags=True):
    """Attach the mandatory CGNS/HDF5 (ADF) attributes to a node.

    Every ADF-H5 node carries ``name``/``label`` (fixed 33-char ASCII), a
    ``type`` data-type string (fixed 3-char ASCII), and an int32 ``flags`` array.
    The root node is the only one without ``flags``.
    """
    node.attrs.create("name", name.encode("ascii"), dtype=_string_dtype(33))
    node.attrs.create("label", label.encode("ascii"), dtype=_string_dtype(33))
    node.attrs.create("type", adf_type.encode("ascii"), dtype=_string_dtype(3))
    if with_flags:
        node.attrs.create("flags", np.array([1], dtype=np.int32))


def _create_node(
    parent,
    name,
    label,
    adf_type="MT",
    data=None,
    compression=None,
    compression_opts=None,
):
    """Create a CGNS node (an HDF5 group) with ADF attributes and optional data.

    ``data`` is stored in the node's ``" data"`` dataset (note the leading space)
    with the numpy dtype that matches ``adf_type``; ``MT`` nodes carry no data.
    """
    node = parent.create_group(name)
    _set_node_attrs(node, name, label, adf_type)
    if adf_type != "MT" and data is not None:
        array = np.asarray(data, dtype=_ADF_DTYPE[adf_type])
        # gzip needs chunked storage, which is impossible for an empty dataset.
        if compression is not None and array.size == 0:
            compression = None
            compression_opts = None
        node.create_dataset(
            " data",
            data=array,
            compression=compression,
            compression_opts=compression_opts,
        )
    return node


def _write_string_node(parent, name, label, text):
    """Write a ``C1`` node whose ``" data"`` holds ``text`` as a char array."""
    return _create_node(parent, name, label, "C1", list(text.encode("ascii")))


def _init_root(f):
    """Write the CGNS/HDF5 root metadata required by the CGNS library.

    This is what makes the file a *conformant* CGNS/HDF5 file (accepted by
    ``cgnscheck``/cgio): the root node attributes, the ``" format"`` and
    ``" hdf5version"`` datasets, and the ``CGNSLibraryVersion`` node.
    """
    import h5py

    _set_node_attrs(
        f, "HDF5 MotherNode", "Root Node of HDF5 File", "MT", with_flags=False
    )
    # The " format" node tells cgio the byte order of the data on disk, which
    # numpy/h5py write in the host's native order.
    data_format = "IEEE_BIG_32" if sys.byteorder == "big" else "IEEE_LITTLE_32"
    f.create_dataset(
        " format",
        data=np.frombuffer((data_format + "\x00").encode("ascii"), dtype=np.int8),
    )
    version = f"HDF5 Version {h5py.version.hdf5_version}".encode("ascii")
    version = version.ljust(33, b"\x00")[:33]
    f.create_dataset(" hdf5version", data=np.frombuffer(version, dtype=np.int8))
    _create_node(f, "CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4", [3.4])


def _write_element_section(
    zone,
    name,
    code,
    elem_range,
    connectivity,
    offsets,
    compression,
    compression_opts,
):
    """Write one ``Elements_t`` section, mirroring what :func:`_read_elements` reads."""
    # Elements_t " data" holds [ElementType, ElementSizeBoundary].
    section = _create_node(zone, name, "Elements_t", "I4", [code, 0])
    _create_node(section, "ElementRange", "IndexRange_t", "I8", elem_range)
    _create_node(
        section,
        "ElementConnectivity",
        "DataArray_t",
        "I8",
        connectivity,
        compression,
        compression_opts,
    )
    if offsets is not None:
        _create_node(
            section,
            "ElementStartOffset",
            "DataArray_t",
            "I8",
            offsets,
            compression,
            compression_opts,
        )


def _write_polyhedra(zone, cells, next_start, compression, compression_opts):
    """Write polyhedra as an NGON_n (faces) + NFACE_n (cells) section pair.

    Each cell is a list (per face) of 0-based node-index arrays. Faces are laid
    out flat in an NGON_n section starting at element number ``next_start`` and
    referenced from the NFACE_n section by their global CGNS element number,
    which is exactly how :func:`_resolve_polyhedra` reads them.
    """
    face_conn = []
    face_offsets = [0]
    cell_refs = []
    cell_offsets = [0]

    face_no = 0
    for faces in cells:
        for face in faces:
            nodes = np.asarray(face, dtype=np.int64)
            face_conn.append(nodes + 1)  # 1-based node indices
            face_offsets.append(face_offsets[-1] + nodes.size)
            cell_refs.append(next_start + face_no)  # global face element number
            face_no += 1
        cell_offsets.append(cell_offsets[-1] + len(faces))

    n_faces = face_no
    n_cells = len(cells)
    ngon_conn = np.concatenate(face_conn) if face_conn else np.empty(0, np.int64)

    _write_element_section(
        zone,
        "NGON_faces",
        NGON_N,
        [next_start, next_start + n_faces - 1],
        ngon_conn,
        np.asarray(face_offsets, dtype=np.int64),
        compression,
        compression_opts,
    )
    nface_start = next_start + n_faces
    _write_element_section(
        zone,
        "NFACE_cells",
        NFACE_N,
        [nface_start, nface_start + n_cells - 1],
        np.asarray(cell_refs, dtype=np.int64),
        np.asarray(cell_offsets, dtype=np.int64),
        compression,
        compression_opts,
    )


def _write_elements(zone, mesh, compression, compression_opts):
    """Write every cell block of ``mesh`` as CGNS ``Elements_t`` sections."""
    next_start = 1  # 1-based CGNS element numbering
    section_id = 0
    polyhedra = []  # accumulated across all polyhedron blocks

    for cell_block in mesh.cells:
        ctype = cell_block.type
        data = cell_block.data

        if ctype.startswith("polyhedron"):
            polyhedra.extend(data)
            continue

        if ctype == "polygon":
            faces = [np.asarray(face, dtype=np.int64) for face in data]
            offsets = np.concatenate([[0], np.cumsum([f.size for f in faces])])
            conn = np.concatenate(faces) if faces else np.empty(0, np.int64)
            n = len(faces)
            section_id += 1
            _write_element_section(
                zone,
                f"NGON_{section_id}",
                NGON_N,
                [next_start, next_start + n - 1],
                conn + 1,  # 1-based
                offsets,
                compression,
                compression_opts,
            )
            next_start += n
            continue

        info = _meshio_to_cgns_type.get(ctype)
        if info is None:
            warn(f'CGNS: unsupported cell type "{ctype}"; block skipped.')
            continue
        code, _ = info
        arr = np.asarray(data, dtype=np.int64)
        n = len(arr)
        section_id += 1
        _write_element_section(
            zone,
            f"{ctype}_{section_id}",
            code,
            [next_start, next_start + n - 1],
            arr.reshape(-1) + 1,  # 1-based
            None,
            compression,
            compression_opts,
        )
        next_start += n

    if polyhedra:
        _write_polyhedra(zone, polyhedra, next_start, compression, compression_opts)


def write(filename, mesh, compression="gzip", compression_opts=4):
    """Write an unstructured mesh to a CGNS/HDF5 file.

    The layout mirrors what :func:`read` expects: every CGNS node is an HDF5
    group carrying a ``label`` attribute, with payloads in ``" data"`` datasets
    (note the leading space). Fixed-size element blocks become ``Elements_t``
    sections; ``polygon`` blocks are written as ``NGON_n`` and ``polyhedron``
    blocks as an ``NGON_n``/``NFACE_n`` pair.
    """
    import h5py

    points = np.asarray(mesh.points, dtype=np.float64)
    n_points, phys_dim = points.shape
    cell_dim = max((cell_block.dim for cell_block in mesh.cells), default=phys_dim)
    # CGNS CellSize counts only the top-dimensional (cell_dim) elements.
    n_cells = sum(
        len(cell_block.data) for cell_block in mesh.cells if cell_block.dim == cell_dim
    )

    with h5py.File(filename, "w") as f:
        _init_root(f)

        # CGNSBase_t " data" holds [CellDimension, PhysicalDimension].
        base = _create_node(f, "Base", "CGNSBase_t", "I4", [cell_dim, phys_dim])

        # Zone_t " data" holds [NVertex, NCell, NBoundVertex] with CGNS shape
        # [IndexDimension=1][3], which cgio stores (dims reversed) as HDF5 (3, 1).
        zone = _create_node(base, "Zone", "Zone_t", "I4", [[n_points], [n_cells], [0]])
        _write_string_node(zone, "ZoneType", "ZoneType_t", "Unstructured")

        grid = _create_node(zone, "GridCoordinates", "GridCoordinates_t")
        coord_names = ["CoordinateX", "CoordinateY", "CoordinateZ"]
        for i in range(phys_dim):
            _create_node(
                grid,
                coord_names[i],
                "DataArray_t",
                "R8",
                np.ascontiguousarray(points[:, i]),
                compression,
                compression_opts,
            )

        _write_elements(zone, mesh, compression, compression_opts)


register_format("cgns", [".cgns"], read, {"cgns": write})
