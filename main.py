from walker import BiomechanicsTools
"""
# --- Options --- #
data_path = "E:\\eWalking_WP1_DataBrut\\OK\\LAO_01\\Venue 2\\AQM\\c3d"
kinematic_model_file_path = "walker\\LAO.bioMod"
static_trial = f"{data_path}\\LAO_01_Statique.c3d"
trials = (f"{data_path}\\LAO_01_Cond0001.c3d",
    f"{data_path}\\LAO_01_Cond0002.c3d",
    f"{data_path}\\LAO_01_Cond0003.c3d",
    f"{data_path}\\LAO_01_Cond0004.c3d",
    f"{data_path}\\LAO_01_Cond0005.c3d",
    f"{data_path}\\LAO_01_Cond0006.c3d",
    f"{data_path}\\LAO_01_Cond0007.c3d",
    f"{data_path}\\LAO_01_Cond0008.c3d",
    f"{data_path}\\LAO_01_Cond0009.c3d",
    f"{data_path}\\LAO_01_Cond0010.c3d",
    f"{data_path}\\LAO_01_Cond0011.c3d",
    f"{data_path}\\LAO_01_Cond0012.c3d",
    f"{data_path}\\LAO_01_Cond0013.c3d",
    f"{data_path}\\LAO_01_Cond0014.c3d",
    f"{data_path}\\LAO_01_Cond0015.c3d",
    f"{data_path}\\LAO_01_Cond0016.c3d")
print(kinematic_model_file_path)
print('****')
# --------------- #


def main():
    print(kinematic_model_file_path)
    # Generate the personalized kinematic model
    tools = BiomechanicsTools(body_mass=58, include_upper_body=True)
    tools.personalize_model(static_trial, kinematic_model_file_path)

    # Perform some biomechanical computation
    for trial in trials:
        print(trial)
        tools.process_trial(trial, compute_automatic_events=False)

    # TODO: Bioviz vizual bug with the end of the trial when resizing the window
    # TODO: Record a tutorial


if __name__ == "__main__":
    main()
"""

import os
from walker import BiomechanicsTools

# --- Options --- #
base_data_path = "E:\\eWalking_WP1_DataBrut\\OK"  # Chemin vers le répertoire de base contenant les sous-dossiers


def process_folder(folder_path, identifier, number, mass):
    # Chemin vers le modèle biomécanique pour cet identifiant
    kinematic_model_file_path = f"walker\\{identifier}.bioMod"

    # Chemin vers le fichier statique et les fichiers d'essai dans le dossier actuel
    static_trial = os.path.join(folder_path, f"{identifier}_{number}_Statique.c3d")
    trials = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.startswith(f"{identifier}_{number}_Cond") and file.endswith(".c3d")
    ]

    if not os.path.exists(static_trial) or not trials:
        print(f"Aucun fichier statique ou de condition trouvé dans {folder_path}")
        return

    # Crée l'outil biomécanique et personnalise le modèle
    tools = BiomechanicsTools(body_mass=mass, sexe='M', include_upper_body=True)
    tools.personalize_model(static_trial, kinematic_model_file_path)

    # Traitement des essais dans le dossier actuel
    for trial in trials:
        print(f"Traitement de {trial}")
        tools.process_trial(trial, compute_automatic_events=False)


def main():
    # Boucle sur tous les sous-dossiers dans le répertoire de base
    all_mass = [71] #58.6, 72, 91.1, 66.7,79.2, 72, 72,47.6, 61, 58,
    ii = 0
    for folder_name in os.listdir(base_data_path):
        if "_" in folder_name:  # Vérifie que le nom de dossier contient un "_"
            identifier, number = folder_name.split("_", 1)
            folder_path = os.path.join(base_data_path, folder_name, "Venue 2", "AQM", "c3d")
            mass = all_mass[ii]
            if os.path.isdir(folder_path):
                print(f"Traitement du dossier : {folder_name}")
                process_folder(folder_path, identifier, number, mass)
            ii = ii+1


if __name__ == "__main__":
    main()
