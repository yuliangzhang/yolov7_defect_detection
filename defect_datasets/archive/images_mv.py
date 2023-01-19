import os,sys
from shutil import copyfile
base_dir = "./images"
image_base_dir = "./images/images/"
file_dir_list = os.listdir(base_dir + "/images")

for file_dir in file_dir_list:

    image_list = os.listdir(os.path.join(image_base_dir, file_dir))

    for image_name in image_list:
        source = os.path.join(image_base_dir, file_dir, image_name)
        target = os.path.join(base_dir, image_name)
        try:
            copyfile(source, target)
        except IOError as e:
            print("Unable to copy file. %s" % e)
            exit(1)
        except:
            print("Unexpected error:", sys.exc_info())
            exit(1)

