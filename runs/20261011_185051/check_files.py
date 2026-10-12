import os
print("Current working directory:", os.getcwd())
print("Files in working directory:", os.listdir('.'))
print("Files in parent directory:", os.listdir('..') if os.path.exists('..') else 'No parent')
