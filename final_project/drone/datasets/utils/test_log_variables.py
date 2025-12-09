"""
Test script to discover available log variables on the Crazyflie.
Run this to see what state variables you can log.
"""

import logging
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

logging.basicConfig(level=logging.ERROR)

group_number = 80
URI = f'radio://0/{group_number}/2M'

print("Connecting to Crazyflie...")
cflib.crtp.init_drivers(enable_debug_driver=False)

with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
    print("Connected!\n")
    
    # Get the log TOC (Table of Contents)
    log_toc = scf.cf.log.toc
    
    print("Available log variables:")
    print("=" * 60)
    
    # Look for state estimation variables
    print("\n📍 STATE ESTIMATION:")
    for group_name in sorted(log_toc.toc.keys()):
        if 'state' in group_name.lower():
            print(f"\n  Group: {group_name}")
            for var_name in sorted(log_toc.toc[group_name].keys()):
                var = log_toc.toc[group_name][var_name]
                print(f"    - {group_name}.{var_name} ({var.ctype})")
    
    # Look for stabilizer variables
    print("\n🎯 STABILIZER (orientation):")
    for group_name in sorted(log_toc.toc.keys()):
        if 'stabilizer' in group_name.lower():
            print(f"\n  Group: {group_name}")
            for var_name in sorted(log_toc.toc[group_name].keys()):
                var = log_toc.toc[group_name][var_name]
                print(f"    - {group_name}.{var_name} ({var.ctype})")
    
    # Look for kalman filter variables
    print("\n🔢 KALMAN FILTER:")
    for group_name in sorted(log_toc.toc.keys()):
        if 'kalman' in group_name.lower():
            print(f"\n  Group: {group_name}")
            for var_name in sorted(log_toc.toc[group_name].keys()):
                var = log_toc.toc[group_name][var_name]
                print(f"    - {group_name}.{var_name} ({var.ctype})")
    
    # Look for motion/velocity variables
    print("\n🚁 MOTION:")
    for group_name in sorted(log_toc.toc.keys()):
        if 'motion' in group_name.lower() or 'velocity' in group_name.lower():
            print(f"\n  Group: {group_name}")
            for var_name in sorted(log_toc.toc[group_name].keys()):
                var = log_toc.toc[group_name][var_name]
                print(f"    - {group_name}.{var_name} ({var.ctype})")

print("\n" + "=" * 60)
print("Done! Use these variable names in your log configuration.")
