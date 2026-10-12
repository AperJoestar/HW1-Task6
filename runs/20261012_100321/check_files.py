import glob
print("Files in working directory:", glob.glob("*"))
print("Files in parent directory:", glob.glob("../*"))
