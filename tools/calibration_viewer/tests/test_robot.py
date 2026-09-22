from calibration_viewer.robot import RobotModel


def test_explicit_mesh_dir_resolves_protomotions_style_path(tmp_path):
    mesh_dir = tmp_path / "assets" / "mesh" / "G1"
    mesh_dir.mkdir(parents=True)
    (mesh_dir / "pelvis.stl").write_text(
        """solid pelvis
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 1 0 0
    vertex 0 1 0
  endloop
endfacet
endsolid pelvis
""",
        encoding="utf-8",
    )
    urdf_dir = tmp_path / "assets" / "urdf" / "for_retargeting"
    urdf_dir.mkdir(parents=True)
    urdf_path = urdf_dir / "robot.urdf"
    urdf_path.write_text(
        """<robot name="test">
  <link name="pelvis">
    <visual><geometry><mesh filename="../mesh/G1/pelvis.stl"/></geometry></visual>
  </link>
</robot>
""",
        encoding="utf-8",
    )

    robot = RobotModel.load(
        urdf_path,
        base_link="pelvis",
        keypoint_links={"pelvis": "pelvis"},
        mesh_dir=mesh_dir,
    )

    assert robot.visuals_loaded
