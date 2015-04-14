#!/usr/bin/env python
import os
import shutil
import urllib
import tarfile
import utils

gtzan_origin = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
gtzan_base_path = os.path.join(os.environ["PYLEARN2_DATA_PATH"], "GTZAN")

if __name__ == '__main__':
    if os.path.isdir(gtzan_base_path):
        if utils.query_yes_no("GTZAN base path %s exists. Do you want to overwrite it? (this will delete all the contents)" % gtzan_base_path):
            shutil.rmtree(gtzan_base_path)
            os.makedirs(gtzan_base_path)
    else:
        print("GTZAN base path %s not found. Creating..." % gtzan_base_path)
        os.makedirs(gtzan_base_path)

    gtzan_dest = os.path.join(gtzan_base_path, os.path.basename(gtzan_origin))
    if os.path.isfile(gtzan_dest):
        if utils.query_yes_no("GTZAN dataset already downloaded in %s. Do you want to download it again?" % gtzan_dest):
            os.remove(gtzan_dest)
            print("Downloading GTZAN dataset from %s to %s" % (gtzan_origin, gtzan_dest))
            urllib.urlretrieve(gtzan_origin, gtzan_dest)
    else:
        print("Downloading GTZAN dataset from %s to %s" % (gtzan_origin, gtzan_dest))
        urllib.urlretrieve(gtzan_origin, gtzan_dest)

    print("Extracting audio files to %s" % gtzan_base_path)
    tar = tarfile.open(gtzan_dest, 'r:gz')
    tar.extractall(gtzan_base_path)
    tar.close()

    # flatten dir structure
    genre_dir = os.path.join(gtzan_base_path, 'genres')
    for genre in os.listdir(genre_dir):
        shutil.move(os.path.join(genre_dir, genre), os.path.join(gtzan_base_path, genre))
    os.rmdir(genre_dir)

    print("All done.")
