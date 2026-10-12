import os
for root, dirs, files in os.walk('../..'):
    if 'SNLS80mV.mat' in files:
        print("Found at:", os.path.abspath(os.path.join(root, 'SNLS80mV.mat')))
