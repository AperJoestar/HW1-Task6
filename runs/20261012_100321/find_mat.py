import os
import shutil

found = False
for root, dirs, files in os.walk('..'):
    if 'SNLS80mV.mat' in files:
        src = os.path.join(root, 'SNLS80mV.mat')
        shutil.copy(src, 'SNLS80mV.mat')
        print(f"Copied from {src} to current directory.")
        found = True
        break

if not found:
    # check parents further up or common locations
    for p in ['../..', '../../..']:
        for root, dirs, files in os.walk(p):
            if 'SNLS80mV.mat' in files:
                src = os.path.join(root, 'SNLS80mV.mat')
                shutil.copy(src, 'SNLS80mV.mat')
                print(f"Copied from {src} to current directory.")
                found = True
                break
        if found: break

if not found:
    print("SNLS80mV.mat not found anywhere!")
