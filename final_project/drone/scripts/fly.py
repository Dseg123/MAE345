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
    #  Load and initialize model(s)
    # ----------------------------------------------------------------------- #
    print("HI")

    # Check if we're using ensemble mode
    use_ensemble = cfg.get('use_ensemble', False)

    if use_ensemble:
        # Load two models for ensemble
        model1 = instantiate(cfg.models)
        model2 = instantiate(cfg.models)

        ckpt_path1 = cfg.ckpt_path1
        ckpt_path2 = cfg.ckpt_path2

        state_dict1 = torch.load(ckpt_path1, map_location="cpu")
        state_dict2 = torch.load(ckpt_path2, map_location="cpu")

        model1.load_state_dict(state_dict1)
        model2.load_state_dict(state_dict2)

        model1.eval()
        model2.eval()

        models = [model1, model2]
        print(f"Loaded ensemble with 2 models from:\n  - {ckpt_path1}\n  - {ckpt_path2}")
    else:
        # Single model mode
        model = instantiate(cfg.models)
        ckpt_path = cfg.ckpt_path
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        models = [model]
        print(f"Loaded single model from: {ckpt_path}")

    # Select device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    for model in models:
        model.to(device)

    # ----------------------------------------------------------------------- #
    #  Set up image transforms
    # ----------------------------------------------------------------------- #

    dataset = CrazyflieILDataset(
        data_dir=cfg.dataset.data_dir,
        image_size=cfg.dataset.image_size,
        augment=cfg.dataset.augment,
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
                if use_ensemble:
                    # Get predictions from both models
                    output1 = models[0](processed_image)
                    output2 = models[1](processed_image)

                    actions1 = models[0].output_to_executable_actions(output1)
                    actions2 = models[1].output_to_executable_actions(output2)

                    # Ensemble strategy: take nonzero action if either model predicts nonzero
                    # This increases sensitivity to obstacles
                    actions = torch.zeros_like(actions1)
                    for action_idx in range(actions1.shape[1]):
                        val1 = actions1[0, action_idx].item()
                        val2 = actions2[0, action_idx].item()

                        # If either model predicts nonzero, use the nonzero value
                        if val1 != 0.0 and val2 != 0.0:
                            # Both nonzero: average them
                            actions[0, action_idx] = (val1 + val2) / 2.0
                        elif val1 != 0.0:
                            # Only model1 nonzero
                            actions[0, action_idx] = val1
                        elif val2 != 0.0:
                            # Only model2 nonzero
                            actions[0, action_idx] = val2
                        else:
                            # Both zero: keep zero
                            actions[0, action_idx] = 0.0

                    print(f"Step {i:03d} actions: {actions} (m1: {actions1}, m2: {actions2})")
                else:
                    # Single model inference
                    output = models[0](processed_image)
                    actions = models[0].output_to_executable_actions(output)
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
        default="train_config_test",
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