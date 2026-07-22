"""Immutable V1 Table001 geometry used by offline calibration."""

V1_TABLE001_BODY_FRAME = "RoboFinals-IKEA-V1:/Root/Table001_01"
V1_TABLE001_BODY_FIDUCIAL_PROVENANCE = (
    "RoboFinals-IKEA-V1 Assets/Table001/Table001.usd /Root/Table001_01 "
    "(PhysicsRigidBodyAPI) and /Root/Table001_01/Sites; V1 assembled-scene "
    "Leg001 shaft endpoints; and exact visual-bound tabletop corners in metres"
)
V1_TABLE001_BODY_FIDUCIALS = (
    {
        "name": "tabletop_xmin_ymin_top_corner",
        "table_point_m": [-0.289993405, -0.209995389, 0.020159841],
        "physical_feature": "Table001 outer visual-bound corner (+local z face)",
    },
    {
        "name": "tabletop_xmin_ymax_top_corner",
        "table_point_m": [-0.289993405, 0.209995389, 0.020159841],
        "physical_feature": "Table001 outer visual-bound corner (+local z face)",
    },
    {
        "name": "tabletop_xmax_ymin_top_corner",
        "table_point_m": [0.289993405, -0.209995389, 0.020159841],
        "physical_feature": "Table001 outer visual-bound corner (+local z face)",
    },
    {
        "name": "tabletop_xmax_ymax_top_corner",
        "table_point_m": [0.289993405, 0.209995389, 0.020159841],
        "physical_feature": "Table001 outer visual-bound corner (+local z face)",
    },
    {
        "name": "tabletop_xmin_ymin_bottom_corner",
        "table_point_m": [-0.289993405, -0.209995389, -0.020159841],
        "physical_feature": "Table001 outer visual-bound corner (-local z face)",
    },
    {
        "name": "tabletop_xmin_ymax_bottom_corner",
        "table_point_m": [-0.289993405, 0.209995389, -0.020159841],
        "physical_feature": "Table001 outer visual-bound corner (-local z face)",
    },
    {
        "name": "tabletop_xmax_ymin_bottom_corner",
        "table_point_m": [0.289993405, -0.209995389, -0.020159841],
        "physical_feature": "Table001 outer visual-bound corner (-local z face)",
    },
    {
        "name": "tabletop_xmax_ymax_bottom_corner",
        "table_point_m": [0.289993405, 0.209995389, -0.020159841],
        "physical_feature": "Table001 outer visual-bound corner (-local z face)",
    },
    {"name": "reg_int1", "table_point_m": [-0.261270000, -0.181380000, -0.000780000]},
    {"name": "reg_int2", "table_point_m": [-0.261270000, 0.181380000, -0.000780000]},
    {"name": "reg_int3", "table_point_m": [0.261360000, -0.181380000, -0.000780000]},
    {"name": "reg_int4", "table_point_m": [0.261360000, 0.181380000, -0.000780000]},
    {
        "name": "leg0_shaft_tip_center",
        "table_point_m": [-0.261231536, -0.181377481, -0.407372223],
        "physical_feature": "Leg001 shaft outer-end centre",
    },
    {
        "name": "leg1_shaft_tip_center",
        "table_point_m": [-0.261231536, 0.181382519, -0.407372223],
        "physical_feature": "Leg001_01 shaft outer-end centre",
    },
    {
        "name": "leg2_shaft_tip_center",
        "table_point_m": [0.261398464, -0.181377481, -0.407372223],
        "physical_feature": "Leg001_03 shaft outer-end centre",
    },
    {
        "name": "leg3_shaft_tip_center",
        "table_point_m": [0.261398464, 0.181382519, -0.407372223],
        "physical_feature": "Leg001_06 shaft outer-end centre",
    },
)
