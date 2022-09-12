from biorbd.model_creation import Axis, BiomechanicalModel, SegmentCoordinateSystem, KinematicChain
import numpy as np


def chord_function(offset, known_center_of_rotation, center_of_rotation_marker, plane_marker):
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
    x = offset * np.sqrt(diameter**2 - offset**2) / diameter
    y = (diameter**2 - offset**2) / diameter

    # project the computed point in the global reference frame
    vect = np.concatenate((x[np.newaxis, :], y[np.newaxis, :], np.zeros((1, n_frames)), np.ones((1, n_frames))))
    return np.einsum('ijk,jk->ik', rt, vect)


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
        return np.einsum('ij,ij->j', v1, v2)

    sp = (point - start_line)[:3, :]
    line = (end_line - start_line)[:3, :]
    return start_line[:3, :] + dot(sp, line) / dot(line, line) * line


class SimplePluginGait(BiomechanicalModel):
    """
    This is the implementation of the Plugin Gait (from Plug-in Gait Reference Guide
    https://docs.vicon.com/display/Nexus212/PDF+downloads+for+Vicon+Nexus)
    """
    def __init__(
        self,
        shoulder_offset: float = None,
        elbow_width: float = None,
        wrist_width: float = None,
        hand_thickness: float = None
    ):
        """
        Parameters
        ----------
        shoulder_offset
            The measured shoulder offset of the subject. If None is provided, it is approximated using
            Rab (2002), A method for determination of upper extremity kinematics
        elbow_width
            The measured width of the elbow. If None is provided 115% of the distance between WRA and WRB is used
        wrist_width
            The measured widht of the wrist. If None is provided, 2cm is used
        hand_thickness
            The measured thickness of the hand. If None is provided, 1cm is used
        """
        super(SimplePluginGait, self).__init__()
        self.shoulder_offset = shoulder_offset
        self.elbow_width = elbow_width
        self.wrist_width = wrist_width
        self.hand_thickness = hand_thickness

        self._define_kinematic_model()
        self._define_dynamic_model()

    def _define_kinematic_model(self):
        # Pelvis is verified
        # Thorax is verified
        # Head is verified
        # Humerus is verified
        # Radius is verified
        # Hand is verified (but is sketchy...)

        self.add_segment(
            "Pelvis",
            translations="xyz",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: (m["LPSI"] + m["RPSI"] + m["LASI"] + m["RASI"]) / 4,
                first_axis=Axis(Axis.Name.X, start=lambda m, kc: (m["LPSI"] + m["RPSI"]) / 2, end="RASI"),
                second_axis=Axis(Axis.Name.Y, start="RASI", end="LASI"),
                axis_to_keep=Axis.Name.Y,
            ),
        )
        # self.add_marker("Pelvis", "SACR", is_technical=False, is_anatomical=True)
        self.add_marker("Pelvis", "LPSI", is_technical=True, is_anatomical=True)
        self.add_marker("Pelvis", "RPSI", is_technical=True, is_anatomical=True)
        self.add_marker("Pelvis", "LASI", is_technical=True, is_anatomical=True)
        self.add_marker("Pelvis", "RASI", is_technical=True, is_anatomical=True)

        self.add_segment(
            "Thorax",
            parent_name="Pelvis",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: m["CLAV"],
                first_axis=Axis(
                    Axis.Name.Z, start=lambda m, kc: (m["T10"] + m["STRN"]) / 2, end=lambda m, kc: (m["C7"] + m["CLAV"]) / 2
                ),
                second_axis=Axis(
                    Axis.Name.X, start=lambda m, kc: (m["T10"] + m["C7"]) / 2, end=lambda m, kc: (m["STRN"] + m["CLAV"]) / 2
                ),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("Thorax", "T10", is_technical=True, is_anatomical=True)
        self.add_marker("Thorax", "C7", is_technical=True, is_anatomical=True)
        self.add_marker("Thorax", "STRN", is_technical=True, is_anatomical=True)
        self.add_marker("Thorax", "CLAV", is_technical=True, is_anatomical=True)
        self.add_marker("Thorax", "RBAK", is_technical=False, is_anatomical=False)

        self.add_segment(
            "Head",
            parent_name="Thorax",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: (m["LFHD"] + m["RFHD"]) / 2,
                first_axis=Axis(
                    Axis.Name.X, start=lambda m, kc: (m["LBHD"] + m["RBHD"]) / 2, end=lambda m, kc: (m["LFHD"] + m["RFHD"]) / 2
                ),
                second_axis=Axis(Axis.Name.Y, start="RFHD", end="LFHD"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        self.add_marker("Head", "LBHD", is_technical=True, is_anatomical=True)
        self.add_marker("Head", "RBHD", is_technical=True, is_anatomical=True)
        self.add_marker("Head", "LFHD", is_technical=True, is_anatomical=True)
        self.add_marker("Head", "RFHD", is_technical=True, is_anatomical=True)

        self.add_segment(
            "RHumerus",
            parent_name="Thorax",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: self._humerus_center_of_rotation(m, kc, "R"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, kc: self._elbow_joint_center(m, kc, "R"),
                    end=lambda m, kc: self._humerus_center_of_rotation(m, kc, "R")
                ),
                second_axis=Axis(
                    Axis.Name.X,
                    start=lambda m, kc: self._wrist_joint_center(m, kc, "R"),
                    end=lambda m, kc: self._elbow_joint_center(m, kc, "R")
                ),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("RHumerus", "RSHO", is_technical=True, is_anatomical=True)
        self.add_marker("RHumerus", "RELB", is_technical=True, is_anatomical=True)
        self.add_marker("RHumerus", "RHUM", is_technical=True, is_anatomical=False)

        self.add_segment(
            "RRadius",
            parent_name="RHumerus",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: self._elbow_joint_center(m, kc, "R"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, kc: self._wrist_joint_center(m, kc, "R"),
                    end=lambda m, kc: self._elbow_joint_center(m, kc, "R")
                ),
                second_axis=Axis(
                    Axis.Name.Y,
                    start=lambda m, kc: kc.segments["RHumerus"].segment_coordinate_system.scs[:, 3, :],
                    end=lambda m, kc: kc.segments["RHumerus"].segment_coordinate_system.scs[:, 1, :],
                ),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("RRadius", "RWRB", is_technical=True, is_anatomical=True)
        self.add_marker("RRadius", "RWRA", is_technical=True, is_anatomical=True)

        self.add_segment(
            "RHand",
            parent_name="RRadius",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: self._hand_origin(m, kc, "R"),
                first_axis=Axis(
                    Axis.Name.Z,
                    start=lambda m, kc: self._hand_origin(m, kc, "R"),
                    end=lambda m, kc: self._wrist_joint_center(m, kc, "R")
                ),
                second_axis=Axis(Axis.Name.Y, start="RWRB", end="RWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("RHand", "RFIN", is_technical=True, is_anatomical=True)

        self.add_segment(
            "LUPPER_ARM",
            parent_name="Thorax",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="LSHO",
                first_axis=Axis(Axis.Name.Z, start="LELB", end="LSHO"),
                second_axis=Axis(Axis.Name.X, start="LWRB", end="LWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("LUPPER_ARM", "LSHO", is_technical=True, is_anatomical=True)
        self.add_marker("LUPPER_ARM", "LELB", is_technical=True, is_anatomical=True)
        self.add_marker("LUPPER_ARM", "LHUM", is_technical=True, is_anatomical=False)

        self.add_segment(
            "LLOWER_ARM",
            parent_name="LUPPER_ARM",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="LELB",
                first_axis=Axis(Axis.Name.Z, start=lambda m, kc: (m["LWRB"] + m["LWRA"]) / 2, end="LELB"),
                second_axis=Axis(Axis.Name.X, start="LWRB", end="LWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("LLOWER_ARM", "LWRB", is_technical=True, is_anatomical=True)
        self.add_marker("LLOWER_ARM", "LWRA", is_technical=True, is_anatomical=True)

        self.add_segment(
            "LHAND",
            parent_name="LLOWER_ARM",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: (m["LWRB"] + m["LWRA"]) / 2,
                first_axis=Axis(Axis.Name.Z, start="LFIN", end=lambda m, kc: (m["LWRB"] + m["LWRA"]) / 2),
                second_axis=Axis(Axis.Name.X, start="LWRB", end="LWRA"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("LHAND", "LFIN", is_technical=True, is_anatomical=True)

        self.add_segment(
            "RTHIGH",
            parent_name="Pelvis",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="RTROC",
                first_axis=Axis(Axis.Name.Z, start="RKNE", end="RTROC"),
                second_axis=Axis(Axis.Name.X, start="RKNM", end="RKNE"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("RTHIGH", "RTROC", is_technical=True, is_anatomical=True)
        self.add_marker("RTHIGH", "RKNE", is_technical=True, is_anatomical=True)
        self.add_marker("RTHIGH", "RKNM", is_technical=False, is_anatomical=True)
        self.add_marker("RTHIGH", "RTHI", is_technical=True, is_anatomical=False)
        self.add_marker("RTHIGH", "RTHID", is_technical=True, is_anatomical=False)

        self.add_segment(
            "RLEG",
            parent_name="RTHIGH",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: (m["RKNM"] + m["RKNE"]) / 2,
                first_axis=Axis(
                    Axis.Name.Z, start=lambda m, kc: (m["RANKM"] + m["RANK"]) / 2, end=lambda m, kc: (m["RKNM"] + m["RKNE"]) / 2
                ),
                second_axis=Axis(Axis.Name.X, start="RKNM", end="RKNE"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        self.add_marker("RLEG", "RANKM", is_technical=False, is_anatomical=True)
        self.add_marker("RLEG", "RANK", is_technical=True, is_anatomical=True)
        self.add_marker("RLEG", "RTIBP", is_technical=True, is_anatomical=False)
        self.add_marker("RLEG", "RTIB", is_technical=True, is_anatomical=False)
        self.add_marker("RLEG", "RTIBD", is_technical=True, is_anatomical=False)

        self.add_segment(
            "RFOOT",
            parent_name="RLEG",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: (m["RANKM"] + m["RANK"]) / 2,
                first_axis=Axis(Axis.Name.X, start="RANKM", end="RANK"),
                second_axis=Axis(Axis.Name.Y, start="RHEE", end="RTOE"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        self.add_marker("RFOOT", "RTOE", is_technical=True, is_anatomical=True)
        self.add_marker("RFOOT", "R5MH", is_technical=True, is_anatomical=True)
        self.add_marker("RFOOT", "RHEE", is_technical=True, is_anatomical=True)

        self.add_segment(
            "LTHIGH",
            parent_name="Pelvis",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin="LTROC",
                first_axis=Axis(Axis.Name.Z, start="LKNE", end="LTROC"),
                second_axis=Axis(Axis.Name.X, start="LKNE", end="LKNM"),
                axis_to_keep=Axis.Name.Z,
            ),
        )
        self.add_marker("LTHIGH", "LTROC", is_technical=True, is_anatomical=True)
        self.add_marker("LTHIGH", "LKNE", is_technical=True, is_anatomical=True)
        self.add_marker("LTHIGH", "LKNM", is_technical=False, is_anatomical=True)
        self.add_marker("LTHIGH", "LTHI", is_technical=True, is_anatomical=False)
        self.add_marker("LTHIGH", "LTHID", is_technical=True, is_anatomical=False)

        self.add_segment(
            "LLEG",
            parent_name="LTHIGH",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: (m["LKNM"] + m["LKNE"]) / 2,
                first_axis=Axis(
                    Axis.Name.Z, start=lambda m, kc: (m["LANKM"] + m["LANK"]) / 2, end=lambda m, kc: (m["LKNM"] + m["LKNE"]) / 2
                ),
                second_axis=Axis(Axis.Name.X, start="LKNE", end="LKNM"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        self.add_marker("LLEG", "LANKM", is_technical=False, is_anatomical=True)
        self.add_marker("LLEG", "LANK", is_technical=True, is_anatomical=True)
        self.add_marker("LLEG", "LTIBP", is_technical=True, is_anatomical=False)
        self.add_marker("LLEG", "LTIB", is_technical=True, is_anatomical=False)
        self.add_marker("LLEG", "LTIBD", is_technical=True, is_anatomical=False)

        self.add_segment(
            "LFOOT",
            parent_name="LLEG",
            rotations="xyz",
            segment_coordinate_system=SegmentCoordinateSystem(
                origin=lambda m, kc: (m["LANKM"] + m["LANK"]) / 2,
                first_axis=Axis(Axis.Name.X, start="LANK", end="LANKM"),
                second_axis=Axis(Axis.Name.Y, start="LHEE", end="LTOE"),
                axis_to_keep=Axis.Name.X,
            ),
        )
        self.add_marker("LFOOT", "LTOE", is_technical=True, is_anatomical=True)
        self.add_marker("LFOOT", "L5MH", is_technical=True, is_anatomical=True)
        self.add_marker("LFOOT", "LHEE", is_technical=True, is_anatomical=True)

    def _define_dynamic_model(self):
        pass

    def _humerus_center_of_rotation(self, m: dict, kc: KinematicChain, side: str) -> np.ndarray:
        """
        This is the implementation of the 'Shoulder joint center, p.69'.

        Parameters
        ----------
        m
            The marker positions in the static
        kc
            The KinematicChain as it is constructed so far
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The position of the origin of the humerus
        """

        thorax_origin = kc.segments["Thorax"].segment_coordinate_system.scs[:, 3, :]
        thorax_x_axis = kc.segments["Thorax"].segment_coordinate_system.scs[:, 0, :]
        thorax_to_sho_axis = m[f"{side}SHO"] - thorax_origin
        shoulder_wand = np.cross(thorax_to_sho_axis[:3, :], thorax_x_axis[:3, :], axis=0)
        shoulder_offset = self.shoulder_offset if self.shoulder_offset is not None else 0.17 * (m[f"{side}SHO"] - m[f"{side}ELB"])[2, :]

        return chord_function(shoulder_offset, thorax_origin, m[f"{side}SHO"], shoulder_wand)

    def _elbow_joint_center(self, m: dict, kc: KinematicChain, side: str) -> np.ndarray:
        """
        Compute the joint center of

        Parameters
        ----------
        m
            The marker positions in the static
        kc
            The KinematicChain as it is constructed so far
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The position of the origin of the elbow
        """

        shoulder_origin = self._humerus_center_of_rotation(m, kc, side)
        elbow_marker = m[f"{side}ELB"]
        wrist_marker = (m[f"{side}WRA"] + m[f"{side}WRB"]) / 2

        elbow_width = self.elbow_width if self.elbow_width is not None else np.linalg.norm(
            m[f"{side}WRA"][:3, :] - m[f"{side}WRB"][:3, :], axis=0
        ) * 1.15
        elbow_offset = elbow_width / 2

        return chord_function(elbow_offset, shoulder_origin, elbow_marker, wrist_marker)

    def _wrist_joint_center(self, m, kc: KinematicChain, side: str) -> np.ndarray:
        """
        Compute the segment coordinate system of the wrist. If wrist_width is not provided 2cm is assumed

        Parameters
        ----------
        m
            The dictionary of marker positions
        kc
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side

        Returns
        -------
        The SCS of the wrist
        """
        elbow_center = self._elbow_joint_center(m, kc, side)
        wrist_bar_center = project_point_on_line(m[f"{side}WRA"], m[f"{side}WRB"], elbow_center)

        offset_axis = np.cross(m[f"{side}WRA"][:3, :] - m[f"{side}WRB"][:3, :], elbow_center[:3, :] - wrist_bar_center, axis=0)
        offset_axis /= np.linalg.norm(offset_axis, axis=0)

        offset = (offset_axis * (self.wrist_width / 2)) if self.wrist_width is not None else 0.02 / 2
        return np.concatenate((wrist_bar_center + offset, np.ones((1, wrist_bar_center.shape[1]))))

    def _hand_origin(self, m, kc: KinematicChain, side: str) -> np.ndarray:
        """
        Compute the origin of the hand. If hand_thickness if not provided, it is assumed to be 1cm

        Parameters
        ----------
        m
            The dictionary of marker positions
        kc
            The kinematic chain as stands at that particular time
        side
            If the markers are from the right ("R") or left ("L") side
        """

        elbow_center = self._elbow_joint_center(m, kc, side)

        wrist_joint_center = self._wrist_joint_center(m, kc, side)
        fin_marker = m[f"{side}FIN"]
        hand_offset = np.repeat(self.hand_thickness / 2 if self.hand_thickness else 0.01 / 2, fin_marker.shape[1])
        wrist_bar_center = project_point_on_line(m[f"{side}WRA"], m[f"{side}WRB"], elbow_center)

        return chord_function(hand_offset, wrist_joint_center, fin_marker, wrist_bar_center)

    @property
    def dof_index(self) -> dict[str, int]:
        return {
            "LHip": 36,  # Left hip flexion
            "LKnee": 39,  # Left knee flexion
            "LAnkle": 42,  # Left ankle flexion
            "LAbsAnkle": 42,  # Left ankle flexion
            "RHip": 27,  # Right hip flexion
            "RKnee": 30,  # Right knee flexion
            "RAnkle": 33,  # Right ankle flexion
            "RAbsAnkle": 33,  # Right ankle flexion
            "LShoulder": 18,  # Left shoulder flexion
            "LElbow": 21,  # Left elbow flexion
            "LWrist": 24,  # Left wrist flexion
            "RShoulder": 9,  # Right shoulder flexion
            "RElbow": 12,  # Right elbow flexion
            "RWrist": 15,  # Right wrist flexion
            "LNeck": None,
            "RNeck": None,
            "LSpine": None,
            "RSpine": None,
            "LHead": None,
            "RHead": None,
            "LThorax": 6,  # Trunk flexion
            "RThorax": 6,  # Trunk flexion
            "LPelvis": 3,  # Pelvis flexion
            "RPelvis": 3,  # Pelvis flexion
        }
