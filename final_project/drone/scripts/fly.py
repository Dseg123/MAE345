import time
import json
import os
from datetime import datetime
import cv2
import argparse
import hydra
from hydra.utils import instantiate

import torch
import torchvision.transforms as transforms

import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

from drone.controls.crazyflie_control import CrazyflieControl
from drone.datasets.dataloader import CrazyflieILDataset

def main(cfg):

    # ----------------------------------------------------------------------- #
    #  Load and initialize model
    # ----------------------------------------------------------------------- #
    print("HI")
    model = instantiate(cfg.models)

    ckpt_path = cfg.ckpt_path
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Select device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # ----------------------------------------------------------------------- #
    #  Set up image transforms
    # ----------------------------------------------------------------------- #

    dataset = CrazyflieILDataset(
        data_dir=cfg.data_dir,
        image_size=cfg.image_size,
        augment=cfg.augment,
    )
    transform = dataset.transform  # match dataset transforms

    # ----------------------------------------------------------------------- #
    #  Set up logging
    # ----------------------------------------------------------------------- #
    # Make a unique directory for this recording session
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.log_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # File where we'll store actions as a single JSON list
    actions_path = os.path.join(save_dir, "actions.json")

    print(f"[LOG] Saving data to: {save_dir}")

    # Will store all records here, then save once at end
    action_records = []

    # ----------------------------------------------------------------------- #
    #  Set up drone
    # ----------------------------------------------------------------------- #
    group_number = cfg.group_number
    URI = f"radio://0/{group_number}/2M"
    cflib.crtp.init_drivers(enable_debug_driver=False)

    with SyncCrazyflie(URI) as scf:
        print("Connected!")
        mc = MotionCommander(scf)
        drone = CrazyflieControl(mc,
                                 camera_index=cfg.camera_index,
                                 )

        print("Hovering...")
        drone.hover(height=cfg.hover_height)

        for i in range(cfg.fly_steps):
            # ========== 1) Get image from drone ==========
            image = drone.get_image(
                crop_top=cfg.crop_top
            )  # H x W x C (uint8)

            # Save raw frame
            img_filename = f"frame_{i:04d}.png"
            img_path = os.path.join(save_dir, img_filename)
            cv2.imwrite(img_path, image)

            # ========== 2) Preprocess ==========
            processed_image = transform(image).unsqueeze(0).to(device)

            # ========== 3) Model inference ==========
            with torch.no_grad():
                output = model(processed_image)

            actions = model.output_to_executable_actions(output)

            print(f"Step {i:03d} actions:", actions)

            # Append record to list
            action_records.append({
                "step": i,
                "image_file": img_filename,
                "actions": actions.tolist(),
            })

            # ========== 4) Send commands ==========
            drone.send_control_commands(actions)
            time.sleep(0.1)

        # Land the drone
        drone.mc.land()

    # ================== SAVE FINAL JSON ==================
    with open(actions_path, "w") as f:
        json.dump(action_records, f, indent=2)

    print("[LOG] Finished run. Data saved to:", save_dir)


if __name__ == "__main__":
    # ----------------------------------------------------------------------- #
    #  Load config and run main
    # ----------------------------------------------------------------------- #
    
    # command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str, 
        required=False,
        default="example_fly_config",
        help="Name of the config file"
    )
    
    # parse arguments
    args = parser.parse_args()
    
    # load the configs from file
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=f"../configs", version_base="1.1")
    cfg = hydra.compose(config_name=args.config_name)
    
    # run the main function
    main(
        cfg=cfg
    )