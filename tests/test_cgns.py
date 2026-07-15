import pytest

import meshlane

from . import helpers


@pytest.mark.parametrize(
    "mesh",
    [
        # helpers.empty_mesh,
        helpers.line_mesh,
        helpers.tri_mesh,
        helpers.tri_mesh_2d,
        helpers.tri_quad_mesh,
        helpers.quad_mesh,
        helpers.quad8_mesh,
        helpers.triangle6_mesh,
        helpers.tet_mesh,
        helpers.tet10_mesh,
        helpers.hex_mesh,
        helpers.hex20_mesh,
        helpers.wedge_mesh,
        helpers.pyramid_mesh,
        helpers.polygon_mesh,
        helpers.polygon_mesh_one_cell,
        helpers.polygon2_mesh,
    ],
)
def test(mesh, tmp_path):
    helpers.write_read(tmp_path, meshlane.cgns.write, meshlane.cgns.read, mesh, 1.0e-15)
