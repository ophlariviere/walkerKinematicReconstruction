from biorbd.model_creation import (
    Axis,
    BiomechanicalModel,
    BiomechanicalModelReal,
    SegmentCoordinateSystem,
    InertiaParameters,
    Mesh,
    Segment,
    Marker,
    Translations,
    Rotations,
)
import numpy as np
import biorbd


def chord_function(offset, known_center_of_rotation, center_of_rotation_marker, plane_marker, direction: int = 1):
    n_frames = offset.shape[0]

    # Create a coordinate system from the markers
    axis1 = plane_marker[:3, :] - known_center_of_rotation[:3, :]
    axis2 = center_of_rotation_marker[:3, :] - known_center_of_rotation[:3, :]
    axis3 = np.cross(axis1, axis2, axis=0)
    axis1 = np.cross(axis2, axis3, axis=0)
    axis1 /= np.linalg.norm(axis1, axis=0)
    axis2 /= np.linalg.norm(axis2, axis=0)
    axis3 /= np.linalg.norm(axis3, axis=0)
    rt = np.identity(4)
    rt = np.repeat(rt, n_frames, axis=1).reshape((4, 4, n_frames))
    rt[:3, 0, :] = axis1
    rt[:3, 1, :] = axis2
    rt[:3, 2, :] = axis3
    rt[:3, 3, :] = known_center_of_rotation[:3, :]

    # The point of interest is the chord from center_of_rotation_marker that has length 'offset' assuming
    # the diameter is the distance between center_of_rotation_marker and known_center_of_rotation.
    # To compute this, project in the rt knowing that by construction, known_center_of_rotation is at 0, 0, 0
    # and center_of_rotation_marker is at a diameter length on y
    diameter = np.linalg.norm(known_center_of_rotation[:3, :] - center_of_rotation_marker[:3, :], axis=0)
    x = offset * direction * np.sqrt(diameter ** 2 - offset ** 2) / diameter
    y = (diameter ** 2 - offset ** 2) / diameter

    # project the computed point in the global reference frame
    vect = np.concatenate((x[np.newaxis, :], y[np.newaxis, :], np.zeros((1, n_frames)), np.ones((1, n_frames))))

    def rt_times_vect(m1, m2):
        return np.einsum("ijk,jk->ik", m1, m2)

    return rt_times_vect(rt, vect)


def point_on_vector(coef: float, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """
    Computes the 3d position of a point using this equation: start + coef * (end - start)

    Parameters
    ----------
    coef
        The coefficient of the length of the segment to use. It is given from the starting point
    start
        The starting point of the segment
    end
        The end point of the segment

    Returns
    -------
    The 3d position of the point
    """

    return start + coef * (end - start)


def project_point_on_line(start_line: np.ndarray, end_line: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Project a point on a line defined by to points (start_line and end_line)

    Parameters
    ----------
    start_line
        The starting point of the line
    end_line
        The ending point of the line
    point
        The point to project

    Returns
    -------
    The projected point
    -------

    """

    def dot(v1, v2):
        return np.einsum("ij,ij->j", v1, v2)

    sp = (point - start_line)[:3, :]
    line = (end_line - start_line)[:3, :]
    return start_line[:3, :] + dot(sp, line) / dot(line, line) * line


def ortho_to_plan(first_pts, second_pts, third_pts):
    vectors_3d = np.cross(
        second_pts[:3, :] - first_pts[:3, :],
        third_pts[:3, :] - first_pts[:3, :],
        axis=0
    )

    # Ajouter la quatrième dimension en copiant celle de `first_pts`
    orthogonal_vectors = np.zeros_like(first_pts)
    orthogonal_vectors[:3, :] = vectors_3d
    orthogonal_vectors[3, :] = first_pts[3, :]  # Copier la 4ème dimension de `first_pts`

    return orthogonal_vectors


class SimplePluginGait(BiomechanicalModel):
    """
    This is the implementation of the Plugin Gait (from Plug-in Gait Reference Guide
    https://docs.vicon.com/display/Nexus212/PDF+downloads+for+Vicon+Nexus)
    """

    def __init__(
            self,
            body_mass: float,
            sexe: str = None,
            shoulder_offset: float = None,
            elbow_width: float = None,
            wrist_width: float = None,
            hand_thickness: float = None,
            leg_length: dict[str, float] = None,
            ankle_width: float = None,
            include_upper_body: bool = True,
    ):
        """
        Parameters
        ----------
        body_mass
            The mass of the full body
        shoulder_offset
            The measured shoulder offset of the subject. If None is provided, it is approximated using
            Rab (2002), A method for determination of upper extremity kinematics
        elbow_width
            The measured width of the elbow. If None is provided 115% of the distance between WRA and WRB is used
        wrist_width
            The measured width of the wrist. If None is provided, 2cm is used
        hand_thickness
            The measured thickness of the hand. If None is provided, 1cm is used
        leg_length
            The measured leg length in a dict["R"] or dict["L"]. If None is provided, the 95% of the ASI height is
            used (therefore assuming the subject is standing upright during the static trial)
        ankle_width
            The measured ankle width. If None is provided, the distance between ANK and HEE is used.
        include_upper_body
            If the upper body should be included in the reconstruction (set all the technical flag of the upper body
            marker false if not included)

        Since more markers are used in our version (namely Knee medial and ankle medial), the KJC and AJC were
        simplified to be the mean of these markers with their respective lateral markers. Hence, 'ankle_width'
        is no more useful
        """
        super(SimplePluginGait, self).__init__()
        self.body_mass = body_mass
        self.sexe = sexe.upper() if sexe else "F"
        self.include_upper_body = include_upper_body
        self.shoulder_offset = shoulder_offset
        self.elbow_width = elbow_width
        self.wrist_width = wrist_width
        self.hand_thickness = hand_thickness
        self.leg_length = leg_length
        self.ankle_width = ankle_width

        self._set_sex_specific_coefficients()
        self._define_kinematic_model()

    def _set_sex_specific_coefficients(self):
        """
        Set the mass distribution coefficients and radii of gyration coefficients
        depending on the sex of the subject.
        """
        if self.sexe == "M":
            self.mass_coefficients = {
                "Pelvis": 0.142,
                "Thorax": 0.333,
                "Head": 0.067,
                "Humerus": 0.024,
                "Radius": 0.017,
                "Hand": 0.006,
                "Femur": 0.123,
                "Tibia": 0.048,
                "Foot": 0.012,
            }
            self.radii_of_gyration = {
                "Pelvis": (1.01, 1.06, 0.95),
                "Thorax": (0.27, 0.25, 0.28),
                "Head": (0.31, 0.25, 0.33),
                "Humerus": (0.31, 0.14, 0.32),
                "Radius": (0.28, 0.11, 0.27),
                "Hand": (0.38, 0.56, 0.22),
                "Femur": (0.29, 0.15, 0.3),
                "Tibia": (0.28, 0.10, 0.28),
                "Foot": (0.17, 0.37, 0.36),
            }
            self.com_scaling = {
                "Head": (-0.062, 0.555, 0.001),
                "Thorax": (-0.036, -0.42, -0.002),
                "Humerus": (0.017, -0.452, -0.026),
                "Radius": (0.01, -0.417, 0.014),
                "Hand": (0.082, -0.839, 0.074),
                "Pelvis": (0.028, -0.28, -0.006),
                "Femur": (-0.041, -0.429, 0.033),
                "Tibia": (-0.048, -0.41, 0.007),
                "Foot": (0.382, -0.151, 0.026),
            }
        else:  # Default to coefficients for females
            self.mass_coefficients = {
                "Pelvis": 0.146,
                "Thorax": 0.304,
                "Head": 0.067,
                "Humerus": 0.022,
                "Radius": 0.013,
                "Hand": 0.005,
                "Femur": 0.146,
                "Tibia": 0.045,
                "Foot": 0.01,
            }
            self.radii_of_gyration = {
                "Pelvis": (0.91, 1.0, 0.79),
                "Thorax": (0.29, 0.27, 0.29),
                "Head": (0.32, 0.27, 0.34),
                "Humerus": (0.33, 0.17, 0.33),
                "Radius": (0.26, 0.14, 0.25),
                "Hand": (0.63, 0.43, 0.58),
                "Femur": (0.31, 0.19, 0.32),
                "Tibia": (0.28, 0.1, 0.28),
                "Foot": (0.17, 0.36, 0.35),
            }
            self.com_scaling = {
                "Head": (-0.07, 0.597, 0.00),
                "Thorax": (-0.016, -0.436, -0.006),
                "Humerus": (-0.073, -0.454, -0.028),
                "Radius": (0.021, -0.411, 0.019),
                "Hand": (0.077, -0.768, 0.048),
                "Pelvis": (-0.009, -0.232, 0.002),
                "Femur": (-0.077, -0.377, 0.009),
                "Tibia": (-0.049, -0.404, 0.031),
                "Foot": (0.27, -0.218, 0.039),
            }

    def _define_kinematic_model(self):
        # Pelvis: verified, The radii of gyration were computed using InterHip normalisation
        # Thorax: verified
        # Head: verified
        # Humerus: verified
        # Radius: verified
        # Hand: Moved the hand joint center to WJC
        # Femur: verified
        # Knee: Used mid-point of 'KNM' and 'KNE' as KJC
        # Ankle: As for knee, we have access to a much easier medial marker (ANKM), so it was used instead
        self["Ground"] = Segment()

        self["Pelvis"] = Segment(
            parent_name="Ground",
            translations=Translations.XYZ,
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=self._pelvis_joint_center,
                first_axis=Axis(name=Axis.Name.Z, start="LASIS", end="RASIS"),
                second_axis=Axis(name=Axis.Name.Y, start="LASIS",
                                 end=lambda m, bio: m["LASIS"] + ortho_to_plan(m["LASIS"], (m["LPSIS"] + m["RPSIS"]) / 2, (m["LASIS"] + m["RASIS"]) / 2)
                                 ),
                axis_to_keep=Axis.Name.Z,
            ),

            mesh=Mesh(("LPSIS", "RPSIS", "RASIS", "LASIS", "LPSIS")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Pelvis"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(self.com_scaling['Pelvis'][1],
                                                              self._pelvis_joint_center(m, bio),
                                                              (self._hip_joint_center(m, bio, 'R') + self._hip_joint_center(m, bio, 'L')) / 2,
                                                              ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Pelvis"] * self.body_mass,
                    coef=self.radii_of_gyration["Pelvis"],
                    start=self._pelvis_joint_center(m, bio),
                    end=(self._hip_joint_center(m, bio, 'L')+self._hip_joint_center(m, bio, 'R'))/2,
                ),
            ),
        )
        # self.add_marker("Pelvis", "SACR", is_technical=True, is_anatomical=True)
        self["Pelvis"].add_marker(Marker("LPSIS", is_technical=True, is_anatomical=True))
        self["Pelvis"].add_marker(Marker("RPSIS", is_technical=True, is_anatomical=True))
        self["Pelvis"].add_marker(Marker("LASIS", is_technical=True, is_anatomical=True))
        self["Pelvis"].add_marker(Marker("RASIS", is_technical=True, is_anatomical=True))

        self["Thorax"] = Segment(
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=self._thorax_joint_center,
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: self._pelvis_joint_center(m, bio),
                    end=lambda m, bio: self._thorax_joint_center(m, bio),
                ),
                second_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, bio: self._pelvis_joint_center(m, bio),
                    end=lambda m, bio: self._pelvis_joint_center(m, bio)
                                       + ortho_to_plan(self._pelvis_joint_center(m, bio), m["SUP"],
                                                       self._thorax_joint_center(m, bio))),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("T10", "C7", "SUP", "STR", "T10")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Thorax"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(self.com_scaling['Thorax'][1],
                                                              self._thorax_joint_center(m, bio),
                                                              self._pelvis_joint_center(m, bio)),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Thorax"] * self.body_mass,
                    coef=self.radii_of_gyration["Thorax"],
                    start=self._thorax_joint_center(m, bio),
                    end=self._pelvis_joint_center(m, bio),
                ),
            ),
        )
        self["Thorax"].add_marker(Marker("T10", is_technical=True, is_anatomical=True))
        self["Thorax"].add_marker(Marker("C7", is_technical=True, is_anatomical=True))
        self["Thorax"].add_marker(Marker("STR", is_technical=True, is_anatomical=True))
        self["Thorax"].add_marker(Marker("SUP", is_technical=True, is_anatomical=True))
        # self["Thorax"].add_marker(Marker("RBAK", is_technical=True, is_anatomical=True))

        self["Head"] = Segment(
            parent_name="Thorax",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=self._thorax_joint_center,
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: self._thorax_joint_center(m, bio),
                    end="HV",
                ),
                second_axis=Axis(Axis.Name.Z,
                                 start=lambda m, bio: self._thorax_joint_center(m, bio),
                                 end=lambda m, bio: self._thorax_joint_center(m, bio) + ortho_to_plan(self._thorax_joint_center(m, bio), m["SEL"], m["HV"])),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(("OCC", "RTEMP", "SEL", "LTEMP", "OCC", "HV")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Head"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(self.com_scaling['Head'][1],
                                                              m["HV"], self._thorax_joint_center(m, bio)
                                                              ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Head"] * self.body_mass,
                    coef=self.radii_of_gyration["Head"],
                    start=m["HV"],
                    end=self._thorax_joint_center(m, bio),
                ),
            ),
        )
        self["Head"].add_marker(Marker("OCC", is_technical=True, is_anatomical=True))
        self["Head"].add_marker(Marker("LTEMP", is_technical=True, is_anatomical=True))
        self["Head"].add_marker(Marker("RTEMP", is_technical=True, is_anatomical=True))
        self["Head"].add_marker(Marker("SEL", is_technical=True, is_anatomical=True))
        self["Head"].add_marker(Marker("HV", is_technical=True, is_anatomical=True))

        self["RHumerus"] = Segment(
            parent_name="Thorax",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._humerus_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._humerus_joint_center(m, bio, "R"),
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: self._humerus_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._humerus_joint_center(m, bio, "R") + ortho_to_plan(self._humerus_joint_center(m, bio, "R"), m["RLHE"], m["RMHE"])
                ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._humerus_joint_center(m, bio, "R"),
                    lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Humerus"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Humerus'][1],
                    start=self._humerus_joint_center(m, bio, "R"),
                    end=self._elbow_joint_center(m, bio, "R")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Humerus"] * self.body_mass,
                    coef=self.radii_of_gyration["Humerus"],
                    start=self._humerus_joint_center(m, bio, "R"),
                    end=self._elbow_joint_center(m, bio, "R"),
                ),
            ),
        )
        self["RHumerus"].add_marker(Marker("RA", is_technical=True, is_anatomical=True))
        self["RHumerus"].add_marker(Marker("RLHE", is_technical=True, is_anatomical=True))
        self["RHumerus"].add_marker(Marker("RMHE", is_technical=True, is_anatomical=True))

        self["RRadius"] = Segment(
            parent_name="RHumerus",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: self._wrist_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._elbow_joint_center(m, bio, "R") + ortho_to_plan(
                        self._elbow_joint_center(m, bio, "R"), m["RUS"], m["RRS"])
                ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._elbow_joint_center(m, bio, "R"),
                    lambda m, bio: self._wrist_joint_center(m, bio, "R"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Radius"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Radius'][1], start=self._elbow_joint_center(m, bio, "R"),
                    end=self._wrist_joint_center(m, bio, "R")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Radius"] * self.body_mass,
                    coef=self.radii_of_gyration["Radius"],
                    start=self._elbow_joint_center(m, bio, "R"),
                    end=self._wrist_joint_center(m, bio, "R"),
                ),
            ),
        )
        self["RRadius"].add_marker(Marker("RUS", is_technical=True, is_anatomical=True))
        self["RRadius"].add_marker(Marker("RRS", is_technical=True, is_anatomical=True))

        self["RHand"] = Segment(
            parent_name="RRadius",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._wrist_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: (m["RHMH2"] + m["RHMH5"]) / 2,
                    end=lambda m, bio: self._wrist_joint_center(m, bio, "R"),
                ),
                second_axis=Axis(Axis.Name.X,
                                 start=lambda m, bio: self._wrist_joint_center(m, bio, "R"),
                                 end=lambda m, bio: self._wrist_joint_center(m, bio, "R") + ortho_to_plan(
                                     self._wrist_joint_center(m, bio, "R"), m["RHMH2"], m["RHMH5"])
                                 ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh((lambda m, bio: self._wrist_joint_center(m, bio, "R"), "RFT3")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients['Hand'] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Hand'][1],
                    start=self._wrist_joint_center(m, bio, "R"),
                    end=(m["RHMH2"] + m["RHMH5"]) / 2,
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Hand"] * self.body_mass,
                    coef=self.radii_of_gyration["Hand"],
                    start=self._wrist_joint_center(m, bio, "R"),
                    end=(m["RHMH2"] + m["RHMH5"]) / 2,
                ),
            ),
        )
        self["RHand"].add_marker(Marker("RFT3", is_technical=True, is_anatomical=True))
        self["RHand"].add_marker(Marker("RHMH2", is_technical=True, is_anatomical=True))
        self["RHand"].add_marker(Marker("RHMH5", is_technical=True, is_anatomical=True))

        self["LHumerus"] = Segment(
            parent_name="Thorax",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._humerus_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._humerus_joint_center(m, bio, "L"),
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: self._humerus_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._humerus_joint_center(m, bio, "L") + ortho_to_plan(
                        self._humerus_joint_center(m, bio, "L"), m["LMHE"], m["LLHE"])
                ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._humerus_joint_center(m, bio, "L"),
                    lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Humerus"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Humerus'][1], start=self._humerus_joint_center(m, bio, "L"), end=self._elbow_joint_center(m, bio, "L")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Humerus"] * self.body_mass,
                    coef=self.radii_of_gyration["Humerus"],
                    start=self._humerus_joint_center(m, bio, "L"),
                    end=self._elbow_joint_center(m, bio, "L"),
                ),
            ),
        )
        self["LHumerus"].add_marker(Marker("LA", is_technical=True, is_anatomical=True))
        self["LHumerus"].add_marker(Marker("LLHE", is_technical=True, is_anatomical=True))
        # TODO: Add ELBM to define the axis
        self["LHumerus"].add_marker(Marker("LMHE", is_technical=True, is_anatomical=True))

        self["LRadius"] = Segment(
            parent_name="LHumerus",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: self._wrist_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._elbow_joint_center(m, bio, "L") + ortho_to_plan(
                        self._elbow_joint_center(m, bio, "L"), m["LRS"], m["LUS"])
                ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._elbow_joint_center(m, bio, "L"),
                    lambda m, bio: self._wrist_joint_center(m, bio, "L"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Radius"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Radius'][1], start=self._elbow_joint_center(m, bio, "L"), end=self._wrist_joint_center(m, bio, "L")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Radius"] * self.body_mass,
                    coef=self.radii_of_gyration["Radius"],
                    start=self._elbow_joint_center(m, bio, "L"),
                    end=self._wrist_joint_center(m, bio, "L"),
                ),
            ),
        )
        self["LRadius"].add_marker(Marker("LUS", is_technical=True, is_anatomical=True))
        self["LRadius"].add_marker(Marker("LRS", is_technical=True, is_anatomical=True))

        self["LHand"] = Segment(
            parent_name="LRadius",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._wrist_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: (m["LHMH2"] + m["LHMH5"]) / 2,
                    end=lambda m, bio: self._wrist_joint_center(m, bio, "L"),
                ),
                second_axis=Axis(Axis.Name.X,
                                 start=lambda m, bio: self._wrist_joint_center(m, bio, "L"),
                                 end=lambda m, bio: self._wrist_joint_center(m, bio, "L") + ortho_to_plan(
                                     self._wrist_joint_center(m, bio, "L"), m["LHMH5"], m["LHMH2"])
                                 ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh((lambda m, bio: self._wrist_joint_center(m, bio, "L"), "LFT3")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Hand"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Hand'][1],
                    start=self._wrist_joint_center(m, bio, "L"),
                    end=(m["LHMH2"] + m["LHMH5"]) / 2,
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Hand"] * self.body_mass,
                    coef=self.radii_of_gyration["Hand"],
                    start=self._wrist_joint_center(m, bio, "L"),
                    end=(m["LHMH2"] + m["LHMH5"]) / 2,
                ),
            ),
        )
        self["LHand"].add_marker(Marker("LFT3", is_technical=True, is_anatomical=True))
        self["LHand"].add_marker(Marker("LHMH2", is_technical=True, is_anatomical=True))
        self["LHand"].add_marker(Marker("LHMH5", is_technical=True, is_anatomical=True))

        self["RFemur"] = Segment(
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._hip_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: self._knee_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._hip_joint_center(m, bio, "R"),
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: self._hip_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._hip_joint_center(m, bio, "R")+ortho_to_plan(self._hip_joint_center(m, bio, "R"), m["RLFE"], m["RMFE"]),
                ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._hip_joint_center(m, bio, "R"),
                    lambda m, bio: self._knee_joint_center(m, bio, "R"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Femur"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Femur'][1], start=self._hip_joint_center(m, bio, "R"), end=self._knee_joint_center(m, bio, "R")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Femur"] * self.body_mass,
                    coef=self.radii_of_gyration["Femur"],
                    start=self._hip_joint_center(m, bio, "R"),
                    end=self._knee_joint_center(m, bio, "R"),
                ),
            ),
        )
        self["RFemur"].add_marker(Marker("RGT", is_technical=True, is_anatomical=True))
        self["RFemur"].add_marker(Marker("RLFE", is_technical=True, is_anatomical=True))
        self["RFemur"].add_marker(Marker("RMFE", is_technical=True, is_anatomical=True))

        self["RTibia"] = Segment(
            parent_name="RFemur",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._knee_joint_center(m, bio, "R"),
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: self._ankle_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._knee_joint_center(m, bio, "R"),
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: self._knee_joint_center(m, bio, "R"),
                    end=lambda m, bio: self._knee_joint_center(m, bio, "R")+ortho_to_plan(self._knee_joint_center(m, bio, "R"), m["RLM"], self._ankle_joint_center(m, bio, "R")),
                ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._knee_joint_center(m, bio, "R"),
                    lambda m, bio: self._ankle_joint_center(m, bio, "R"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Tibia"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Tibia'][1], start=self._knee_joint_center(m, bio, "R"), end=self._ankle_joint_center(m, bio, "R")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Tibia"] * self.body_mass,
                    coef=self.radii_of_gyration["Tibia"],
                    start=self._knee_joint_center(m, bio, "R"),
                    end=self._ankle_joint_center(m, bio, "R"),
                ),
            ),
        )
        self["RTibia"].add_marker(Marker("RLM", is_technical=True, is_anatomical=True))
        self["RTibia"].add_marker(Marker("RSPH", is_technical=True, is_anatomical=True))
        self["RTibia"].add_marker(Marker("RATT", is_technical=True, is_anatomical=True))

        self["RFoot"] = Segment(
            parent_name="RTibia",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._ankle_joint_center(m, bio, "R"),
                first_axis=Axis(Axis.Name.X, start="RCAL", end=lambda m, bio: (m['RMFH1']+m['RMFH5'])/2),
                second_axis=Axis(Axis.Name.Y, start="RCAL", end=lambda m, bio: m["RCAL"]+ortho_to_plan(m["RCAL"], m["RMFH5"], m["RMFH1"])),
                axis_to_keep=Axis.Name.X,
            ),
            mesh=Mesh(("RTT2", "RMFH5", "RLM", "RCAL", "RSPH", "RMFH1", "RTT2")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Foot"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Foot'][1], start=self._ankle_joint_center(m, bio, "R"), end=(m['RMFH1']+m['RMFH5'])/2
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Foot"] * self.body_mass,
                    coef=self.radii_of_gyration["Foot"],
                    start=self._ankle_joint_center(m, bio, "R"),
                    end=(m['RMFH1']+m['RMFH5'])/2,
                ),
            ),
        )
        self["RFoot"].add_marker(Marker("RTT2", is_technical=True, is_anatomical=True))
        self["RFoot"].add_marker(Marker("RMFH5", is_technical=True, is_anatomical=True))
        self["RFoot"].add_marker(Marker("RCAL", is_technical=True, is_anatomical=True))
        # self["RFoot"].add_marker(Marker("RLM", is_technical=True, is_anatomical=True))
        # self["RFoot"].add_marker(Marker("RSPH", is_technical=True, is_anatomical=True))
        self["RFoot"].add_marker(Marker("RMFH1", is_technical=True, is_anatomical=True))

        self["LFemur"] = Segment(
            parent_name="Pelvis",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._hip_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: self._knee_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._hip_joint_center(m, bio, "L"),
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: self._hip_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._hip_joint_center(m, bio, "L") + ortho_to_plan(
                        self._hip_joint_center(m, bio, "L"), m["LMFE"], m["LLFE"]),
                ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._hip_joint_center(m, bio, "L"),
                    lambda m, bio: self._knee_joint_center(m, bio, "L"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Femur"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Femur'][1], start=self._hip_joint_center(m, bio, "L"), end=self._knee_joint_center(m, bio, "L")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Femur"] * self.body_mass,
                    coef=self.radii_of_gyration["Femur"],
                    start=self._hip_joint_center(m, bio, "L"),
                    end=self._knee_joint_center(m, bio, "L"),
                ),
            ),
        )
        self["LFemur"].add_marker(Marker("LGT", is_technical=True, is_anatomical=True))
        self["LFemur"].add_marker(Marker("LLFE", is_technical=True, is_anatomical=True))
        self["LFemur"].add_marker(Marker("LMFE", is_technical=True, is_anatomical=True))

        self["LTibia"] = Segment(
            parent_name="LFemur",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._knee_joint_center(m, bio, "L"),
                first_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, bio: self._ankle_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._knee_joint_center(m, bio, "L"),
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, bio: self._knee_joint_center(m, bio, "L"),
                    end=lambda m, bio: self._knee_joint_center(m, bio, "L") + ortho_to_plan(
                        self._knee_joint_center(m, bio, "L"), self._ankle_joint_center(m, bio, "L"), m["LLM"]),
                ),
                axis_to_keep=Axis.Name.Y,
            ),
            mesh=Mesh(
                (
                    lambda m, bio: self._knee_joint_center(m, bio, "L"),
                    lambda m, bio: self._ankle_joint_center(m, bio, "L"),
                )
            ),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Tibia"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Tibia'][1], start=self._knee_joint_center(m, bio, "L"), end=self._ankle_joint_center(m, bio, "L")
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Tibia"] * self.body_mass,
                    coef=self.radii_of_gyration["Tibia"],
                    start=self._knee_joint_center(m, bio, "L"),
                    end=self._ankle_joint_center(m, bio, "L"),
                ),
            ),
        )
        self["LTibia"].add_marker(Marker("LLM", is_technical=True, is_anatomical=True))
        self["LTibia"].add_marker(Marker("LSPH", is_technical=True, is_anatomical=True))
        self["LTibia"].add_marker(Marker("LATT", is_technical=True, is_anatomical=True))

        self["LFoot"] = Segment(
            parent_name="LTibia",
            rotations=Rotations.XYZ,
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, bio: self._ankle_joint_center(m, bio, "L"),
                first_axis=Axis(Axis.Name.X, start="LCAL", end=lambda m, bio: (m['LMFH1'] + m['LMFH5']) / 2),
                second_axis=Axis(Axis.Name.Y, start="LCAL",
                                 end=lambda m, bio: m["LCAL"] + ortho_to_plan(m["LCAL"], m["LMFH1"], m["LMFH5"])),
                axis_to_keep=Axis.Name.X,
            ),
            mesh=Mesh(("LTT2", "LMFH5", "LLM", "LCAL", "LSPH", "LMFH1", "LTT2")),
            inertia_parameters=InertiaParameters(
                mass=lambda m, bio: self.mass_coefficients["Foot"] * self.body_mass,
                center_of_mass=lambda m, bio: point_on_vector(
                    self.com_scaling['Foot'][1], start=self._ankle_joint_center(m, bio, "L"),
                    end=(m['LMFH1'] + m['LMFH5']) / 2
                ),
                inertia=lambda m, bio: InertiaParameters.radii_of_gyration_to_inertia(
                    mass=self.mass_coefficients["Foot"] * self.body_mass,
                    coef=self.radii_of_gyration["Foot"],
                    start=self._ankle_joint_center(m, bio, "L"),
                    end=(m['LMFH1'] + m['LMFH5']) / 2,
                ),
            ),
        )
        self["LFoot"].add_marker(Marker("LTT2", is_technical=True, is_anatomical=True))
        self["LFoot"].add_marker(Marker("LMFH5", is_technical=True, is_anatomical=True))
        self["LFoot"].add_marker(Marker("LCAL", is_technical=True, is_anatomical=True))
        # self["LFoot"].add_marker(Marker("LLM", is_technical=True, is_anatomical=True))
        # self["LFoot"].add_marker(Marker("LSPH", is_technical=True, is_anatomical=True))
        self["LFoot"].add_marker(Marker("LMFH1", is_technical=True, is_anatomical=True))

    def _lumbar_5(self, m, bio):
        right_hip = self._hip_joint_center(m, bio, "R")
        left_hip = self._hip_joint_center(m, bio, "L")
        return np.nanmean((left_hip, right_hip), axis=0) + np.array((0.0, 0.0, 0.828, 0))[:, np.newaxis] * np.repeat(
            np.linalg.norm(left_hip - right_hip, axis=0)[np.newaxis, :], 4, axis=0
        )

    def _pelvis_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        Pelvis_scaling = {'Lumbar': {'F': [-0.34, 0.049, 0], 'M': [-0.335, -0.032, 0]}}
        sex = self.sexe

        LASIS = m["LASIS"]
        RASIS = m["RASIS"]
        MASIS = (m["RASIS"] + m["LASIS"]) / 2
        MPSIS = (m["RPSIS"] + m["LPSIS"]) / 2

        Zaxis = (RASIS[:3, :]-LASIS[:3, :])
        Zaxis = Zaxis / np.linalg.norm(Zaxis, axis=0)
        Yaxis = np.cross(Zaxis, MASIS[:3, :]-MPSIS[:3, :], axis=0)
        Yaxis = Yaxis / np.linalg.norm(Yaxis, axis=0)
        Xaxis = np.cross(Yaxis, Zaxis, axis=0)
        Xaxis = Xaxis / np.linalg.norm(Xaxis, axis=0)

        pelvis_width = np.mean(np.linalg.norm(LASIS[:3] - RASIS[:3], axis=0))

        Lumbar_JC = np.zeros_like(LASIS)
        Lumbar_JC[:3, :] = MASIS[:3, :] + pelvis_width * (
                    Xaxis * Pelvis_scaling['Lumbar'][sex][0] + Yaxis * Pelvis_scaling['Lumbar'][sex][1] + Zaxis *
                    Pelvis_scaling['Lumbar'][sex][2])
        Lumbar_JC[3, :] = LASIS[3, :]

        return Lumbar_JC

    def _pelvis_center_of_mass(self, m: dict, bio: BiomechanicalModelReal) -> np.ndarray:
        """
        This computes the center of mass of the thorax

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        """
        sex = self.sexe

        LASIS = m["LASIS"]
        RASIS = m["RASIS"]
        MASIS = (m["RASIS"] + m["LASIS"]) / 2
        MPSIS = (m["RPSIS"] + m["LPSIS"]) / 2

        # Pelvis axis
        Zaxis = (RASIS[:3, :] - LASIS[:3, :])
        Zaxis = Zaxis / np.linalg.norm(Zaxis, axis=0)
        Yaxis = np.cross(Zaxis, MASIS[:3, :] - MPSIS[:3, :], axis=0)
        Yaxis = Yaxis / np.linalg.norm(Yaxis, axis=0)
        Xaxis = np.cross(Yaxis, Zaxis, axis=0)
        Xaxis = Xaxis / np.linalg.norm(Xaxis, axis=0)

        # Pelvis origin
        PelvisOrigin = self._pelvis_joint_center(m, bio)
        HipJointCenter = self._hip_joint_center(m, bio, 'R')

        # PelvisLength
        PelvisOrigin = np.mean(np.linalg.norm(PelvisOrigin[:2] - HipJointCenter[:2], axis=0))

        PelvisCoM = np.zeros_like(LASIS)
        PelvisCoM[:3, :] = PelvisOrigin[:3, :] + self.com_scaling['Pelvis'][0] * Xaxis + self.com_scaling['Pelvis'][1] * Yaxis + self.com_scaling['Pelvis'][2] * Zaxis
        PelvisCoM[3, :] = LASIS[3, :]
        return PelvisCoM

    def _thorax_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        Trunk_angles = {'Thorax': {'F': 92 * np.pi / 180, 'M': 94 * np.pi / 180},
                        'Cervical': {'F': 14 * np.pi / 180, 'M': 8 * np.pi / 180},
                        'Shoulder': {'F': -5 * np.pi / 180, 'M': -11 * np.pi / 180}}  # in radian
        Trunk_scaling = {'Thorax': {'F': 0.50, 'M': 0.52},
                         'Cervical': {'F': 0.53, 'M': 0.55},
                         'Shoulder': {'F': 0.36, 'M': 0.33}}

        sex = self.sexe

        lumbar_JC = self._pelvis_joint_center(m, bio)
        Xaxis = (m["SUP"][:3, :] - m["C7"][:3, :])
        Xaxis = Xaxis / np.linalg.norm(Xaxis, axis=0)
        Zaxis = np.cross(Xaxis, m["C7"][:3, :] - lumbar_JC[:3, :], axis=0)
        Zaxis = Zaxis / np.linalg.norm(Zaxis, axis=0)
        Yaxis = np.cross(Zaxis, Xaxis, axis=0)
        Yaxis = Yaxis / np.linalg.norm(Yaxis, axis=0)

        thorax_width = np.mean(np.linalg.norm(m["SUP"][:3, :] - m["C7"][:3, :], axis=0))

        Cervical_JC = np.zeros_like(m["LASIS"])
        angle = Trunk_angles['Cervical'][sex]
        Cervical_JC[:3, :] = m['C7'][:3, :] + thorax_width * Trunk_scaling['Cervical'][sex] * (
                    np.cos(angle) * Xaxis + np.sin(angle) * Yaxis)
        Cervical_JC[3, :] = m['C7'][3, :]
        return Cervical_JC

    def _thorax_center_of_mass(self, m: dict, bio: BiomechanicalModelReal) -> np.ndarray:
        """
        This computes the center of mass of the thorax

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        """
        com = point_on_vector(0.63, start=m["C7"], end=self._lumbar_5(m, bio))
        com[0, :] = self._thorax_joint_center(m, bio)[0, :]  # Make sur the center of mass is symmetric
        return com

    def _head_joint_center(self, m: dict, bio: BiomechanicalModelReal):
        return (m["LTEMP"] + m["RTEMP"]) / 2

    def _head_center_of_mass(self, m: dict, bio: BiomechanicalModelReal):
        return point_on_vector(
            0.52,
            start=m["SEL"],
            end=m["OCC"],
        )

    def _humerus_joint_center(self, m: dict, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        This is the implementation of the 'Shoulder joint center, p.69'.

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The position of the origin of the humerus
        """

        Trunk_angles = {'Thorax': {'F': 92 * np.pi / 180, 'M': 94 * np.pi / 180},
                        'Cervical': {'F': 14 * np.pi / 180, 'M': 8 * np.pi / 180},
                        'Shoulder': {'F': -5 * np.pi / 180, 'M': -11 * np.pi / 180}}  # in radian
        Trunk_scaling = {'Thorax': {'F': 0.50, 'M': 0.52},
                         'Cervical': {'F': 0.53, 'M': 0.55},
                         'Shoulder': {'F': 0.36, 'M': 0.33}}

        sex = self.sexe

        lumbar_JC = self._pelvis_joint_center(m, bio)
        Xaxis = (m["SUP"][:3, :] - m["C7"][:3, :])
        Xaxis = Xaxis / np.linalg.norm(Xaxis, axis=0)
        Zaxis = np.cross(Xaxis, m["C7"][:3, :] - lumbar_JC[:3, :], axis=0)
        Zaxis = Zaxis / np.linalg.norm(Zaxis, axis=0)
        Yaxis = np.cross(Zaxis, Xaxis, axis=0)
        Yaxis = Yaxis / np.linalg.norm(Yaxis, axis=0)

        thorax_width = np.mean(np.linalg.norm(m["SUP"][:3, :] - m["C7"][:3, :], axis=0))

        Shoulder_JC = np.zeros_like(m["LASIS"])
        angle = Trunk_angles['Shoulder'][sex]
        Shoulder_JC[:3, :] = m[side + 'A'][:3, :] + thorax_width * Trunk_scaling['Shoulder'][
                sex] * (np.cos(angle) * Xaxis + np.sin(angle) * Yaxis)
        Shoulder_JC[3, :] = m['C7'][3, :]

        return Shoulder_JC

    def _elbow_joint_center(self, m: dict, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the joint center of

        Parameters
        ----------
        m
            The marker positions in the static
        bio
            The BiomechanicalModelReal as it is constructed so far
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The position of the origin of the elbow
        """

        shoulder_origin = self._humerus_joint_center(m, bio, side)
        elbow_marker = (m[f"{side}LHE"] + m[f"{side}MHE"]) / 2
        wrist_marker = (m[f"{side}RS"] + m[f"{side}US"]) / 2

        elbow_width = (
            self.elbow_width
            if self.elbow_width is not None
            else np.linalg.norm(m[f"{side}RS"][:3, :] - m[f"{side}US"][:3, :], axis=0) * 1.15
        )
        elbow_offset = elbow_width / 2

        return elbow_marker # chord_function(elbow_offset, shoulder_origin, elbow_marker, wrist_marker)

    def _wrist_joint_center(self, m, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the segment coordinate system of the wrist. If wrist_width is not provided 2cm is assumed

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The SCS of the wrist
        """

        elbow_center = self._elbow_joint_center(m, bio, side)
        wrist_bar_center = project_point_on_line(m[f"{side}RS"], m[f"{side}US"], elbow_center)
        offset_axis = np.cross(
            m[f"{side}RS"][:3, :] - m[f"{side}US"][:3, :], elbow_center[:3, :] - wrist_bar_center, axis=0
        )
        offset_axis /= np.linalg.norm(offset_axis, axis=0)

        offset = (offset_axis * (self.wrist_width / 2)) if self.wrist_width is not None else 0.02 / 2
        return (m[f"{side}RS"]+ m[f"{side}US"])/2 # np.concatenate((wrist_bar_center - offset, np.ones((1, wrist_bar_center.shape[1]))))  #wrist_bar_center + offset

    def _hand_center(self, m, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the origin of the hand. If hand_thickness if not provided, it is assumed to be 1cm

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """

        elbow_center = self._elbow_joint_center(m, bio, side)
        wrist_joint_center = self._wrist_joint_center(m, bio, side)
        fin_marker = m[f"{side}FT3"]
        hand_offset = np.repeat(self.hand_thickness / 2 if self.hand_thickness else 0.01 / 2, fin_marker.shape[1])
        wrist_bar_center = project_point_on_line(m[f"{side}RS"], m[f"{side}US"], elbow_center)

        return chord_function(hand_offset, wrist_joint_center, fin_marker, wrist_bar_center)

    def _legs_length(self, m, bio: BiomechanicalModelReal):
        # TODO: Verify 95% makes sense
        return {
            "R": self.leg_length["R"] if self.leg_length else np.nanmean(
                np.linalg.norm(m["RGT"][:3, :] - m["RLM"][:3, :], axis=0)),
            "L": self.leg_length["L"] if self.leg_length else np.nanmean(
                np.linalg.norm(m["LGT"][:3, :] - m["LLM"][:3, :], axis=0)),
        }

    def _hip_joint_center(self, m, bio: BiomechanicalModelReal, side: str) -> np.ndarray:
        """
        Compute the hip joint center. The LegLength is not provided, the height of the TROC is used (therefore assuming
        the subject is standing upright during the static trial)

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """

        Pelvis_scaling = {'R_hip': {'F': [-0.139, -0.336, 0.372], 'M': [-0.095, -0.37, 0.361]},
                          'L_hip': {'F': [-0.139, -0.336, -0.372], 'M': [-0.095, -0.37, -0.361]}}
        sex = self.sexe

        LASIS = m["LASIS"]
        RASIS = m["RASIS"]
        MASIS = (m["RASIS"] + m["LASIS"]) / 2
        MPSIS = (m["RPSIS"] + m["LPSIS"]) / 2

        Zaxis = (RASIS[:3, :] - LASIS[:3, :])
        Zaxis = Zaxis / np.linalg.norm(Zaxis, axis=0)
        Yaxis = np.cross(Zaxis, MASIS[:3, :] - MPSIS[:3, :], axis=0)
        Yaxis = Yaxis / np.linalg.norm(Yaxis, axis=0)
        Xaxis = np.cross(Yaxis, Zaxis, axis=0)
        Xaxis = Xaxis / np.linalg.norm(Xaxis, axis=0)

        pelvis_width = np.mean(np.linalg.norm(LASIS[:3] - RASIS[:3], axis=0))

        Hip_JC = np.zeros_like(LASIS)
        Hip_JC[:3, :] = MASIS[:3, :] + pelvis_width * (
                Xaxis * Pelvis_scaling[f"{side}_hip"][sex][0] + Yaxis * Pelvis_scaling[f"{side}_hip"][sex][1] + Zaxis *
                Pelvis_scaling[f"{side}_hip"][sex][2])
        Hip_JC[3, :] = LASIS[3, :]
        return Hip_JC

    def _knee_axis(self, side) -> Axis:
        """
        Define the knee axis

        Parameters
        ----------
        side
            If the markers are from the right ("R") or left ("L") side
        """
        if side == "R":
            return Axis(Axis.Name.Y, start=f"{side}LFE", end=f"{side}MFE")
        elif side == "L":
            return Axis(Axis.Name.Y, start=f"{side}MFE", end=f"{side}LFE")
        else:
            raise ValueError("side should be 'R' or 'L'")

    def _knee_joint_center(self, m, bio: BiomechanicalModelReal, side) -> np.ndarray:
        """
        Compute the knee joint center. This is a simplified version since the KNM exists

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """
        return (m[f"{side}MFE"] + m[f"{side}LFE"]) / 2

    def _ankle_joint_center(self, m, bio: BiomechanicalModelReal, side) -> np.ndarray:
        """
        Compute the ankle joint center. This is a simplified version sie ANKM exists

        Parameters
        ----------
        m
            The dictionary of marker positions
        bio
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """

        return (m[f"{side}SPH"] + m[f"{side}LM"]) / 2

    @property
    def dof_index(self) -> dict[str, tuple[int, ...]]:
        """
        Returns a dictionary with all the dof to export to the C3D and their corresponding XYZ values in the generalized
        coordinate vector
        """

        # TODO: Some of these values as just copy of their relative
        return {"LHip": (36, 37, 38),
                "LKnee": (39, 40, 41),
                "LAnkle": (42, 43, 44),
                "LAbsAnkle": (42, 43, 44),
                "LFootProgress": (42, 43, 44),
                "RHip": (27, 28, 29),
                "RKnee": (30, 31, 32),
                "RAnkle": (33, 34, 35),
                "RAbsAnkle": (33, 34, 35),
                "RFootProgress": (33, 34, 35),
                "LShoulder": (18, 19, 20),
                "LElbow": (21, 22, 23),
                "LWrist": (24, 25, 26),
                "RShoulder": (9, 10, 11),
                "RElbow": (12, 13, 14),
                "RWrist": (15, 16, 17),
                "LNeck": None,
                "RNeck": None,
                "LSpine": None,
                "RSpine": None,
                "LHead": None,
                "RHead": None,
                "LThorax": (6, 7, 8),
                "RThorax": (6, 7, 8),
                "LPelvis": (3, 4, 5),
                "RPelvis": (3, 4, 5),
                }
