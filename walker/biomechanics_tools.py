import itertools
import os.path
from scipy.signal import butter, filtfilt, savgol_filter
import biorbd
from biorbd.model_creation import C3dData
import bioviz
import ezc3d
import numpy as np
from scipy import signal
from .misc import differentiate, to_rotation_matrix, to_euler
from .plugin_gait import SimplePluginGait
from shapely.geometry import Point, Polygon



def suffix_to_all(values: tuple[str, ...] | list[str, ...], suffix: str) -> tuple[str, ...]:
    return tuple(f"{n}{suffix}" for n in values)


class BiomechanicsTools:
    def __init__(self, body_mass: float, sexe: str, include_upper_body: bool = True):
        self.generic_model = SimplePluginGait(body_mass, sexe, include_upper_body=True)
        self.model = None
        # biorbd.Model("C:\\Users\\felie\\PycharmProjects\\walkerKinematicReconstruction\\walker\\49mks.bioMod")

        self.is_kinematic_reconstructed: bool = False
        self.is_inverse_dynamic_performed: bool = False
        self.c3d_path: str | None = None
        self.c3d: ezc3d.c3d | None = None
        self.t: np.ndarray = np.ndarray(())
        self.q: np.ndarray = np.ndarray(())
        self.qdot: np.ndarray = np.ndarray(())
        self.qddot: np.ndarray = np.ndarray(())
        self.tau: np.ndarray = np.ndarray(())
        self.center_of_mass = np.ndarray(())
        self.force = np.ndarray(())
        self.cop = np.ndarray(())
        self.moment = np.ndarray(())

        self.events = None
        self.bioviz_window: bioviz.Viz | None = None

    @property
    def is_model_loaded(self):
        return self.model is not None

    def _compute_center_of_mass(self) -> np.ndarray:
        if not self.is_model_loaded:
            raise RuntimeError("The biorbd model must be loaded. You can do so by calling generate_personalized_model")

        self.com = np.array([self.model.CoM(q).to_array() for q in self.q.T]).T
        return self.com

    @staticmethod
    def nan_filtfilt(b, a, data):
        """ Applique filtfilt en ignorant les NaN """
        nan_mask = np.isnan(data)
        if np.all(nan_mask):  # Si toute la colonne est NaN, on la met à zéro
            return np.zeros_like(data)

        filtered = np.copy(data)
        valid_idx = np.where(~nan_mask)[0]  # Indices des valeurs non-NaN
        if len(valid_idx) > 1:  # Filtrer uniquement si au moins 2 points valides
            filtered[valid_idx] = filtfilt(b, a, data[valid_idx])

        return filtered


    def forcedatafilter(self, data, order, sampling_rate, cutoff_freq):
        nyquist = 0.5 * sampling_rate
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_data = np.empty([len(data[:, 0]), len(data[0, :])])
        for ii in range(3):
            # filtered_data[ii, :] = medfilt(data[ii, :], kernel_size=5)
            filtered_data[ii, :] = self.nan_filtfilt(b, a, data[ii, :])
        return filtered_data

    @staticmethod
    def normalize(v):
        """ Normalize a vector. """
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def personalize_model(self, static_trial: str, model_path: str = "temporary.bioMod"):
        """
        Collapse the generic model according to the data of the static trial

        Parameters
        ----------
        static_trial
            The path of the c3d file of the static trial to create the model from
        model_path
            The path of the generated bioMod file
        """
        data=C3dData(static_trial)
        n_frames = data.values["LASIS"].shape

        if len(n_frames) == 1:  # Si on a une seule frame
            for key in data.values:  # Parcourir chaque clé du dictionnaire
                data.values[key] = data.values[key][:, np.newaxis]  # Ajouter une dimension pour obtenir la forme (4, 1)
        self.generic_model.write(save_path=model_path, data=data)
        self.model = biorbd.Model(model_path)

    def process_trial(self, trial: str, compute_automatic_events: bool = False) -> None:
        """
        Performs everything to do with a specific trial, including kinematic reconstruction and export

        Parameters
        ----------
        trial
            The path to the c3d file of the trial to reconstruct the kinematics from
        compute_automatic_events
            If the automatic event finding algorithm should be used. Otherwise, the events in the c3d file are used
        """

        self.process_kinematics(trial, visualize=False)
        self.inverse_dynamics()
        self.calculate_H()
        # Write the c3d as if it was the plug in gate output
        path = os.path.dirname(trial)
        file_name = os.path.splitext(os.path.basename(trial))[0]
        self.to_c3d(f"{path}/{file_name}_processed7.c3d", compute_automatic_events=compute_automatic_events)

    def process_kinematics(self, trial: str, visualize: bool = False):
        """
        Performs the kinematics reconstruction

        Parameters
        ----------
        trial
            The path to the c3d file of the trial to reconstruct the kinematics from
        visualize
            If the reconstruction should be shown

        Returns
        -------
        This method populates self.c3d, self.c3d_path, self.t, self.q, self.qdot and self.qddot
        """
        self.load_c3d_file(trial)
        # frames = self._select_frames_to_reconstruct(acceptance_threshold=0.7)
        n_frames = self.c3d["data"]["points"].shape[2]
        frames = slice(0, n_frames)
        self.reconstruct_kinematics(frames=frames)
        self.unwrap_kinematics()
        """
        if visualize is True:
            self.show_kinematic_reconstruction()
            """

    def load_c3d_file(self, trial):
        """
        Load the c3d in the variables

        Parameters
        ----------
        trial
            The path to the c3d file of the trial to reconstruct the kinematics from
        """
        self.c3d_path = trial
        self.c3d = ezc3d.c3d(self.c3d_path, extract_forceplat_data=True)

    def _select_frames_to_reconstruct(self, acceptance_threshold: float = 0.7) -> slice:
        """
        Select the first and last frame where the percentage threshold of visible markers is satisfied

        Parameters
        ----------
        acceptance_threshold
            The percentage of markers that must be present. Default all the marker should be seen

        Returns
        -------
        The frames to reconstruct
        """
        technical_markers = tuple(name.to_string() for name in self.model.technicalMarkerNames())
        n_technical_markers = len(technical_markers)
        c3d_marker_names = self.c3d["parameters"]["POINT"]["LABELS"]["value"]
        marker_index = tuple(c3d_marker_names.index(n) for n in technical_markers)
        n_frames = self.c3d["data"]["points"].shape[2]

        n_nan_marker_per_frame = np.sum(
            np.isnan(np.sum(self.c3d["data"]["points"][:, marker_index, :], axis=0)), axis=0
        )
        missing_percentage_per_frame = n_nan_marker_per_frame / n_technical_markers

        is_frame_accepted = list(missing_percentage_per_frame < (1 - acceptance_threshold))
        first_frame = is_frame_accepted.index(True)
        is_frame_accepted.reverse()
        last_frame = n_frames - is_frame_accepted.index(True)

        return slice(first_frame, last_frame)

    def reconstruct_kinematics(self, frames: slice = slice(None)) -> np.ndarray:
        """
        Reconstruct the kinematics of the specified trial assuming a biorbd model is loaded using a Kalman filter

        Parameters
        ----------
        frames
            The frames to reconstruct

        Returns
        -------
        The matrix nq x ntimes of the reconstructed kinematics
        """
        if not self.is_model_loaded:
            raise RuntimeError("The biorbd model must be loaded. You can do so by calling generate_personalized_model")
        """
        first_frame_c3d = self.c3d["header"]["points"]["first_frame"]
        last_frame_c3d = self.c3d["header"]["points"]["last_frame"]
        
        n_frames_before = 0#(frames.start - first_frame_c3d) if frames.start is not None else 0
        n_frames_after = 0#(last_frame_c3d - frames.stop + 1) if frames.stop is not None else 0
        n_frames_total = last_frame_c3d - first_frame_c3d + 1
        """

        self.model.segment(10).range = [[biorbd.Range(-np.pi*40 / 180, np.pi*40 / 180)],
                                        [biorbd.Range(-np.pi*180 / 180, np.pi*180 / 180)],
                                        [biorbd.Range(-np.pi*40 / 180, np.pi*40 / 180)]]
        self.model.segment(11).range = [[biorbd.Range(-np.pi*60/180, np.pi*60 / 180)],
                                        [biorbd.Range(-np.pi*180 / 180, np.pi*180 / 180)],
                                        [biorbd.Range(-np.pi*60 / 180, np.pi*60/ 180)]]
        self.model.segment(12).range = [[biorbd.Range(-np.pi*15 / 180, np.pi*15 / 180)],
                                        [biorbd.Range(-np.pi*90 / 180, np.pi*90 / 180)],
                                        [biorbd.Range(-np.pi*20 / 180, np.pi*20 / 180)]]
        self.model.segment(13).range = [[biorbd.Range(-np.pi * 40 / 180, np.pi * 40 / 180)],
                                        [biorbd.Range(-np.pi * 180 / 180, np.pi * 180 / 180)],
                                        [biorbd.Range(-np.pi * 40 / 180, np.pi * 40 / 180)]]
        self.model.segment(14).range = [[biorbd.Range(-np.pi * 60 / 180, np.pi * 60 / 180)],
                                        [biorbd.Range(-np.pi * 180 / 180, np.pi * 180 / 180)],
                                        [biorbd.Range(-np.pi * 60 / 180, np.pi * 60 / 180)]]
        self.model.segment(15).range = [[biorbd.Range(-np.pi * 15 / 180, np.pi * 15 / 180)],
                                        [biorbd.Range(-np.pi * 90 / 180, np.pi * 90 / 180)],
                                        [biorbd.Range(-np.pi * 20 / 180, np.pi * 20 / 180)]]
        c3d = ezc3d.c3d(self.c3d_path, extract_forceplat_data=True)
        labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
        data = c3d["data"]["points"]
        n_frames = data[:, :, frames].shape[2]
        marker_names = tuple(n.to_string() for n in self.model.technicalMarkerNames())
        index_in_c3d = np.array(tuple(labels.index(name) if name in labels else -1 for name in marker_names))
        markers_in_c3d = np.ndarray((3, len(index_in_c3d), n_frames)) * np.nan
        markers_in_c3d[:, index_in_c3d >= 0, :] = data[:3, index_in_c3d[index_in_c3d >= 0], frames] / 1000  # To meter
        ik = biorbd.InverseKinematics(self.model, markers_in_c3d)
        ik.solve(method="trf")
        self.q = ik.q

        # self.t, self.q, self.qdot, self.qddot = biorbd.extended_kalman_filter(self.model, self.c3d_path, frames=frames)

        # Align the data with the c3d
        """
        n_q = self.q.shape[0]
        dof_padding_before = np.zeros((n_q, n_frames_before))
        dof_padding_after = np.zeros((n_q, n_frames_after))
        self.t = np.linspace(first_frame_c3d, last_frame_c3d, n_frames_total)
        self.q = np.concatenate((dof_padding_before, self.q, dof_padding_after), axis=1)
        self.qdot = np.concatenate((dof_padding_before, self.qdot, dof_padding_after), axis=1)
        self.qddot = np.concatenate((dof_padding_before, self.qddot, dof_padding_after), axis=1)
        """
        self.is_kinematic_reconstructed = True
        return self.q

    def calculate_H(self):
        n = self.q.shape[1]
        angular_momentum = np.zeros((3, n))
        for ii in range(n):
            angular_momentum[:, ii] = self.model.angularMomentum(self.q[:, ii], self.qdot[:, ii], True).to_array()
        self.H = angular_momentum
        return self.H

    def relative_to_vertical(self, segment: str, angle_sequence: str, q: np.ndarray = None) -> np.array:
        """
        Provide the Euler angles of the specified segment relative to vertical

        Parameters
        ----------
        segment
            The name of the segment to express relative to vertical
        angle_sequence
            The sequence of angle to reconstruct to
        q
            The generalized coordinates to use. If None is sent, then self.q is
            used (assuming kinematics was reconstructed)

        Returns
        -------
        The Euler angles for the specified segment
        """
        if q is None and not self.is_kinematic_reconstructed:
            raise RuntimeError("The kinematics must be reconstructed before performing the unwrap")
        q = self.q if q is None else q

        segment_idx = tuple(s.name().to_string() for s in self.model.segments()).index(segment)
        jcs = np.ndarray((4, 4, self.q.shape[1]))
        for i, q in enumerate(q.T):
            jcs[:, :, i] = self.model.globalJCS(q, segment_idx).to_array()
        return to_euler(jcs, angle_sequence)

    def unwrap_kinematics(self):
        """
        Performs unwrap on the kinematics from which it re-expressed in terms of matrix rotation before
        (which makes it more likely to be in the same quadrant)

        Returns
        -------

        """

        if not self.is_kinematic_reconstructed:
            raise RuntimeError("The kinematics must be reconstructed before performing the unwrap")

        segment_names = tuple(s.name().to_string() for s in self.model.segments())
        dof_names = tuple(n.to_string() for n in self.model.nameDof())
        for segment_name in segment_names:
            segment = self.model.segment(segment_names.index(segment_name))
            angle_sequence = segment.seqR().to_string()
            if not angle_sequence:
                continue
            angle_names = tuple(
                segment.nameDof(i).to_string()
                for i in range(segment.nbDofTrans(), segment.nbDofTrans() + segment.nbDofRot())
            )
            angle_index = tuple(dof_names.index(f"{segment_name}_{angle_name}") for angle_name in angle_names)

            data = self.q[angle_index, :]
            rot = to_rotation_matrix(angles=data, angle_sequence=angle_sequence)
            # euler_angles = to_euler(rot, angle_sequence)
            #  self.q[angle_index, :] = self.wrap_to_180(np.unwrap(euler_angles, axis=1))
            self.q[angle_index, :] = np.unwrap(to_euler(rot, angle_sequence), axis=1)

    def wrap_to_180(self, angles):
        return (angles + 180) % 360 - 180


    def get_cycles(self, side) -> tuple[int, ...]:
        """
        Get the cycles slices based on the C3D file. More specifically,
        it returns the all the indices of the Foot Strikes

        Parameters
        ----------
        side
            The side ("right" or "left") to get the slices from

        Returns
        -------
        All the cycles
        """
        if not self.c3d_path:
            raise RuntimeError("A C3D file must be loaded")

        if side != "Right" and side != "Left":
            raise ValueError("side must be 'Right' or 'Left'")

        events_side = self.c3d["parameters"]["EVENT"]["CONTEXTS"]["value"]
        events_tag = self.c3d["parameters"]["EVENT"]["LABELS"]["value"]
        events_time = self.c3d["parameters"]["EVENT"]["TIMES"]["value"][1, :]

        rate = self.c3d["header"]["points"]["frame_rate"]
        first_time = self.c3d["header"]["points"]["first_frame"] / rate
        last_time = self.c3d["header"]["points"]["last_frame"] / rate
        t = np.linspace(first_time, last_time, self.c3d["data"]["points"].shape[2])
        events_index = [list(t > event).index(True) for event in events_time]

        out = []
        for event_side, event_tag, event_index in zip(events_side, events_tag, events_index):
            if event_side != side:
                continue
            if event_tag != "Foot Strike":
                continue
            out.append(event_index)

        return tuple(out)

    def show_kinematic_reconstruction(self):
        """
        Opens a BioViz window and show the kinematic reconstruction
        """

        if not self.is_kinematic_reconstructed:
            raise RuntimeError("The kinematics must be reconstructed before showing the reconstruction")

        viz = bioviz.Viz(loaded_model=self.model)
        viz.load_movement(self.q)
        viz.load_experimental_markers(self.c3d_path)
        viz.radio_c3d_editor_model.click()
        viz.exec()

    def inverse_dynamics(self) -> np.ndarray:
        """
        Perform the inverse dynamics of a previously reconstructed kinematics.

        This function calculates the generalized forces (tau) using the model's kinematics and external forces from
        force platforms.


        Returns
        -------
        np.ndarray
            The generalized forces (tau) calculated from the inverse dynamics.
        """
        if not self.is_kinematic_reconstructed:
            raise RuntimeError("The kinematics must be reconstructed before performing inverse dynamics")

        # contact_names = ["LFoot", "RFoot"]

        num_contacts = len(self.c3d['data']['platform'])
        num_frames = len(self.q[0, :])
        fsPF = self.c3d['parameters']['ANALOG']['RATE']['value'][0]
        units = self.c3d['parameters']['POINT']['UNITS']['value'][0]
        fsMks = self.c3d['parameters']['POINT']['RATE']['value'][0]
        sampling_factor = int(fsPF/fsMks)

        labels = self.c3d['parameters']['POINT']['LABELS']['value']
        rcal_index = labels.index('RCAL')
        rcal_position = ((self.c3d['data']['points'][:3, rcal_index, :]/1000) if units == 'mm'
                         else self.c3d['data']['points'][:3, rcal_index, :])
        lcal_index = labels.index('LCAL')
        lcal_position = ((self.c3d['data']['points'][:3, lcal_index, :]/1000) if units == 'mm'
                         else self.c3d['data']['points'][:3, lcal_index, :])

        # Initialize arrays for storing external forces and moments
        force_filtered = np.zeros((num_contacts, 3, sampling_factor * num_frames))
        moment_filtered = np.zeros((num_contacts, 3, sampling_factor * num_frames))
        cop_filtered = np.zeros((num_contacts, 3, sampling_factor * num_frames))
        moment_origin_adjusted = np.zeros((num_contacts, 3, sampling_factor * num_frames))
        platform_origin = np.zeros((num_contacts, 3, 1))

        # Process force platform data
        for contact_idx in range(len(self.c3d['data']['platform'])):
            platform_data = self.c3d['data']['platform'][contact_idx]

            force_filtered[contact_idx] = self.forcedatafilter(platform_data['force'], 4, fsPF, 10)
            moment_filtered[contact_idx] = self.forcedatafilter(
                (platform_data['moment'] / 1000) if units == 'mm' else platform_data['moment'],
                order=4, sampling_rate=fsPF, cutoff_freq=10)
            cop_filtered[contact_idx] = self.forcedatafilter(
                (platform_data['center_of_pressure'] / 1000) if units == 'mm' else platform_data['center_of_pressure'],
                order=4, sampling_rate=fsPF, cutoff_freq=10
            )
            platform_origin[contact_idx, :, 0] = (
                np.mean(platform_data['corners'], axis=1) / 1000) if units == 'mm' else np.mean(
                platform_data['corners'], axis=1)

            """
                for frame in range(sampling_factor * num_frames):
                r = platform_origin[contact_idx, :, 0] - cop_filtered[contact_idx, :, frame]
                moment_offset = np.cross(r, force_filtered[contact_idx, :, frame])
                moment_origin_adjusted[contact_idx, :, frame] = moment_filtered[contact_idx, :, frame] + moment_offset
            """
        # Assign the adjusted moments back to the filtered array
        # moment_filtered = moment_origin_adjusted

        # Initialize arrays for storing forces and points of application
        self.force = np.empty((num_contacts, 9, num_frames))
        point_application = np.zeros((num_contacts, 3, num_frames))

        # Apply Savitzky-Golay filter for smoothing the position data
        self.q = savgol_filter(self.q, 31, 3, axis=1)

        # Initialize arrays for angular velocity and acceleration
        angular_velocity = np.gradient(self.q, axis=1, edge_order=2) / (1 / fsMks)
        angular_acceleration = np.gradient(angular_velocity, axis=1, edge_order=2) / (1 / fsMks)

        # Assign filtered kinematic data
        self.qdot = angular_velocity
        self.qddot = angular_acceleration

        tau_data = []

        # Perform inverse dynamics frame-by-frame
        for i in range(num_frames):
            self.ext_load = self.model.externalForceSet()

            for contact_idx in range(len(self.c3d['data']['platform'])):
                if force_filtered[contact_idx, 2, sampling_factor * i] > 20:
                    # Extract the spatial vector (moment and force) in the global frame
                    spatial_vector = np.concatenate((moment_filtered[contact_idx, :, sampling_factor * i],
                                                     force_filtered[contact_idx, :, sampling_factor * i]))

                    # Define the application point in the global frame
                    point_application[contact_idx, :, i] = cop_filtered[contact_idx, :, sampling_factor * i]
                    point_app_global = platform_origin[contact_idx, :, 0]  # point_application[contact_idx, :, i]
                    corners = ((self.c3d['data']['platform'][contact_idx]['corners']/1000) if units == 'mm'
                               else (self.c3d['data']['platform'][contact_idx]['corners']))
                    sommets = [(corners[0, 0], corners[1, 0], corners[2, 0]), (corners[0, 1], corners[1, 1], corners[2, 1]),
                               (corners[0, 2], corners[1, 2], corners[2, 2]), (corners[0, 3], corners[1, 3], corners[2, 3])]
                    polygone = Polygon(sommets)
                    Lcal_is_PF = polygone.contains(Point(lcal_position[0, i], lcal_position[1, i]))
                    if Lcal_is_PF:
                        closest_segment = "LFoot"
                    else:
                        closest_segment = "RFoot"

                    name = biorbd.String(closest_segment)
                    self.ext_load.add(name, spatial_vector, point_app_global)

                # Store the force data in self.force for later use or analysis
                self.force[contact_idx, 0:3, i] = point_application[contact_idx, :, i]
                self.force[contact_idx, 3:6, i] = force_filtered[contact_idx, :, sampling_factor * i]
                self.force[contact_idx, 6:, i] = moment_filtered[contact_idx, :, sampling_factor * i]

            # Calculate the inverse dynamics tau using the model's function
            tau = self.model.InverseDynamics(self.q[:, i], self.qdot[:, i], self.qddot[:, i], self.ext_load)
            tau_data.append(tau.to_array())
            self.ext_load = []  # Reset external load for the next iteration

        # Convert tau_data to numpy array and store the results
        tau_data = np.array(tau_data)
        self.tau = tau_data.T  # Transpose to match expected output format

        self.is_inverse_dynamic_performed = True
        return self.tau

    def find_feet_events(self) -> tuple[int, tuple[str, ...], tuple[str, ...], np.ndarray]:
        """
        Returns
        -------
        number of events: int
            The number of events
        event_contexts: tuple[str, ...]
            If a specific event arrived the on 'Left' or on the 'Right'
        event_labels: tuple[str, ...]
            If a specific event is a 'Foot Strike' or a Foot Off'
        event_times: np.ndarray
            The time for a specific event. the first row should be all zeros for some unknown reason
        """

        def find_foot_events(heel_marker_name: str, toe_marker_name: str):
            """
            Finds the events where the foot interacts with the ground. The return is expected to have an equal number
            of heel strikes and toe off.

            The algorithm is to take the lowest velocity of the heel, then to find the first time this velocity hits 0;
            this is the heel strike.
            The maximum heel velocity after that point is prior to the toe off and the highest toe velocity is just
            after. The toe off is therefore 80% of that distance towards the max velocity

            Parameters
            ----------
            heel_marker_name
                The name of the heel marker in the model
            toe_marker_name
                The name of the toe marker in the model

            Returns
            -------
            The 'heel strikes' and 'toe off' event. The number of each is expected to be equal
            """

            markers = biorbd.markers_to_array(self.model, self.q)
            heel_idx = biorbd.marker_index(self.model, heel_marker_name)
            toe_idx = biorbd.marker_index(self.model, toe_marker_name)
            heel_height = markers[(2,), heel_idx, :]
            heel_velocity = differentiate(heel_height, self.t[1] - self.t[0])
            toe_height = markers[(2,), toe_idx, :]
            toe_velocity = differentiate(toe_height, self.t[1] - self.t[0])
            toe_acceleration = differentiate(toe_velocity, self.t[1] - self.t[0])

            idx_peaks_heel_strike = []
            idx_peak_pre_heel_strike = signal.find_peaks(-heel_velocity[0, :], height=0.5)[0]
            for heel_idx in idx_peak_pre_heel_strike:
                # find the first time the signal crosses 0 from that lowest point
                idx_peaks_heel_strike.append(heel_idx + np.argmax(np.diff(np.sign(heel_velocity[:, heel_idx:])) != 0))

            idx_peaks_toe_off = []
            t_peaks_pre_toe_off = signal.find_peaks(heel_velocity[0, :], height=0.5)[0]
            for toe_idx in t_peaks_pre_toe_off:
                # find the first time the signal crosses 0 from that lowest point
                idx_peaks_post_toe_off = np.argmax(np.diff(np.sign(toe_acceleration[:, toe_idx:])) != 0)
                idx_peaks_toe_off.append(int(toe_idx + 0.8 * idx_peaks_post_toe_off))

            # Associate each heel strike with its toe off
            first_toe_off_idx = -1
            for i, toe in enumerate(idx_peaks_toe_off):
                if toe > idx_peaks_heel_strike[0]:
                    first_toe_off_idx = i
                    break
            last_heel_strike_idx = -1
            for i, heel in enumerate(reversed(idx_peaks_heel_strike)):
                if heel < idx_peaks_toe_off[-1]:
                    last_heel_strike_idx = len(idx_peaks_heel_strike) - i
                    break

            if first_toe_off_idx == -1 or last_heel_strike_idx == -1:
                Warning("No heel strikes that correspond to the toe offs were found")

            heel_strikes = idx_peaks_heel_strike[:last_heel_strike_idx]
            toe_off = idx_peaks_toe_off[first_toe_off_idx:]
            if len(heel_strikes) != len(toe_off):
                Warning("The number of heel strikes and toe off does not match")

            return heel_strikes, toe_off

        left_foot_events = find_foot_events("LHEE", "LTOE")
        right_foot_events = find_foot_events("RHEE", "RTOE")
        # From that point, it is assumed that `len(events[0]) == len(events[1])`, that is there are equal number of
        # foot strikes and toe off

        events_number = (len(left_foot_events[0]) + len(right_foot_events[0])) * 2
        events_contexts = ("Left",) * len(left_foot_events[0]) * 2 + ("Right",) * len(right_foot_events[0]) * 2
        events_labels = ("Foot Strike", "Foot Off") * int(events_number / 2)
        events_times = np.array(
            (
                (0.0,) * events_number,
                self.t[
                    np.array(
                        tuple(
                            itertools.chain(  # flatten the left/right, heel strike/toe off
                                *[[heel, toe] for heel, toe in zip(*left_foot_events)]
                                + [[heel, toe] for heel, toe in zip(*right_foot_events)]
                            )
                        )
                    )
                ],
            )
        )
        return events_number, events_contexts, events_labels, events_times

    def _dispatch_events_from_bioviz(self):
        self.events = [self.bioviz_window.n_events]
        self.events += self.bioviz_window.analyses_c3d_editor.convert_event_for_c3d(
            self.c3d["header"]["points"]["frame_rate"]
        )

        self.bioviz_window.vtk_window.close()

    def to_c3d(self, save_path: str, compute_automatic_events: bool = False) -> None:
        """
        Create a Nexus-like c3d file from the reconstructed kinematics

        Parameters
        ----------
        save_path
            The path where to save the c3d
        compute_automatic_events
            If the automatic event finding algorithm should be used. Otherwise, the events in the c3d file are used
        """

        if not self.is_kinematic_reconstructed:
            raise RuntimeError(
                "Kinematics should be reconstructed before writing to c3d. " "Please call 'kinematic_reconstruction'"
            )

        c3d = ezc3d.c3d()

        # Fill it with points, angles, power, force, moment
        c3d["parameters"]["POINT"]["RATE"]["value"] = [int(self.c3d["parameters"]["POINT"]["RATE"]["value"][0])]
        c3d.add_parameter("POINT", "ANGLE_UNITS", ["deg"])
        point_names = [name.to_string() for name in self.model.markerNames()]
        point_names.extend(["CentreOfMass", "CentreOfMassFloor", "CoP1", "CoP2", "force1",
                            "force2", "moment1", "moment2", "H"])
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Angles"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Vitesse"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Acc"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Power"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Force"))
        point_names.extend(suffix_to_all(tuple(self.generic_model.dof_index.keys()), "Moment"))
        c3d["parameters"]["POINT"]["UNITS"] = self.c3d["parameters"]["POINT"]["UNITS"]

        # Transfer the marker data to the new c3d
        c3d["parameters"]["POINT"]["LABELS"]["value"] = point_names
        first_frame = self.c3d["header"]["points"]["first_frame"]
        last_frame = self.c3d["header"]["points"]["last_frame"]
        n_frame = last_frame - first_frame + 1
        data = np.ndarray((4, len(point_names), n_frame)) * np.nan
        data[3, ...] = 1
        for i, name_in_c3d in enumerate(self.c3d["parameters"]["POINT"]["LABELS"]["value"]):
            if name_in_c3d[0] == "*" or name_in_c3d not in point_names:
                continue
            # Make sure it is in the right order
            data[:, point_names.index(name_in_c3d), :] = self.c3d["data"]["points"][:, i, :]

        self._compute_center_of_mass()
        data[:3, point_names.index("CentreOfMass"), :] = self.com
        data[:3, point_names.index("CentreOfMassFloor"), :] = self.com
        data[2, point_names.index("CentreOfMassFloor"), :] = 0
        data[:3, point_names.index("H"), :] = self.H
        data[:3, point_names.index("CoP1"), :] = self.force[0, :3, :]
        data[:3, point_names.index("CoP2"), :] = self.force[1, :3, :]
        data[:3, point_names.index("force1"), :] = self.force[0, 3:6, :]
        data[:3, point_names.index("force2"), :] = self.force[1, 3:6, :]
        data[:3, point_names.index("moment1"), :] = self.force[0, 6:, :]
        data[:3, point_names.index("moment2"), :] = self.force[1, 6:, :]


        # Dispatch the kinematics and kinematics
        for dof, idx in self.generic_model.dof_index.items():
            if idx is None:
                continue
            data[:3, point_names.index(f"{dof}Angles"), :] = self.q[idx, :] * 180 / np.pi
            data[:3, point_names.index(f"{dof}Vitesse"), :] = self.qdot[idx, :]
            data[:3, point_names.index(f"{dof}Acc"), :] = self.qddot[idx, :]
            data[:3, point_names.index(f"{dof}Moment"), :] = self.tau[idx, :]
            data[:3, point_names.index(f"{dof}Power"), :] = self.tau[idx, :] * self.qdot[idx, :]

        c3d["data"]["points"] = data

        """
        # Affichage
        self.bioviz_window = bioviz.Viz(loaded_model=self.model)
        self.bioviz_window.load_movement(self.q)
        self.bioviz_window.load_experimental_forces(self.force[:, :6, :], segments=['Ground', 'Ground'],
                                                    normalization_ratio=0.5)
        self.bioviz_window.load_experimental_markers(self.c3d_path)
        self.bioviz_window.radio_c3d_editor_model.click()

        if compute_automatic_events:
            self.bioviz_window.clear_events()
            # Find and add events
            self.events = self.find_feet_events()
            events_number, events_contexts, events_labels, events_times = self.events
            for context, label, time in zip(events_contexts, events_labels, events_times[1, :]):
                frame = (
                    int(time * self.c3d["header"]["points"]["frame_rate"]) - self.c3d["header"]["points"]["first_frame"]
                )
                self.bioviz_window.set_event(frame, f"{context} {label}")
        self.bioviz_window.analyses_c3d_editor.export_c3d_button.disconnect()
        self.bioviz_window.analyses_c3d_editor.export_c3d_button.clicked.connect(self._dispatch_events_from_bioviz)
        self.bioviz_window.exec()
        if self.events is None:
            raise RuntimeError("No events found, have you clicked Export C3D?")
        events_number, events_contexts, events_labels, events_times = self.events
        
        c3d.add_parameter("EVENT", "USED", (events_number,))
        c3d.add_parameter("EVENT", "CONTEXTS", events_contexts)
        c3d.add_parameter("EVENT", "LABELS", events_labels)
        c3d.add_parameter("EVENT", "TIMES", events_times)
"""
        # Copy the header
        for element in self.c3d["header"]:
            for item in self.c3d["header"][element]:
                c3d["header"][element][item] = self.c3d["header"][element][item]

        # Dispatch the analog and force_plate data
        c3d["parameters"]["ANALOG"] = self.c3d["parameters"]["ANALOG"]
        for i, label in enumerate(self.c3d["parameters"]["ANALOG"]["LABELS"]["value"]):
            label = label.replace("Force.", "")
            label = label.replace("Moment.", "")
            c3d["parameters"]["ANALOG"]["LABELS"]["value"][i] = label

        c3d["parameters"]["FORCE_PLATFORM"] = self.c3d["parameters"]["FORCE_PLATFORM"]
        c3d["data"]["analogs"] = self.c3d["data"]["analogs"]

        # Write the data
        c3d.write(save_path)
