import os
import sys
import matplotlib.image as mpimg
import glob
import collections
import yaml
import numpy as np

BASE_DIR = "/Users/dewalgupta/Documents/ucsd/lab"
Pose = collections.namedtuple('Pose', ['timestamp', 'pose'])

class dataloader(object):
    def __init__(self):
        raise NotImplementedError

    def preprocess(self):
        raise NotImplementedError

    def get_data(self):
        # read ground truth
        with open(self.gt_file, 'r') as f:
            tmp = f.readlines()
            tmp = [t.strip().split(" ") for t in tmp]
            self.gt = [Pose(float(l[0]), [float(x) for x in l[1:]]) for l in tmp]

        # load times
        with open(self.ts_file, 'r') as f:
            self.times = f.readlines()
            self.times = [float(t) for t in self.times]

        # load images
        self.image_fnames = sorted(glob.glob(os.path.join(self.fr_path, "*."+self.img_ftype)))
        self.len = len(self.image_fnames)
        self.images = {}
        
        with open(self.settings_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            cx = self.config['Camera.cx']
            cy = self.config['Camera.cy']
            fx = self.config['Camera.fx']
            fy = self.config['Camera.fy']
            
            self.K = np.array([[fx, 0.0, cx, 0.0], 
                               [0.0, fy, cy, 0.0], 
                               [0.0, 0.0, 1.0, 0.0]])
                
                
    def get_frame(self, tstamp):
        if tstamp not in self.times:
            print("Timestamp not found in dataset, could not retrieve image.")

        idx = self.times.index(tstamp)
        return self.get_frame_idx(idx)
    

    def get_frame_idx(self, idx):
        if idx >= len(self.image_fnames) or idx < 0:
            print("Bad index")

        if idx not in self.images.keys():
            self.images[idx] = mpimg.imread(self.image_fnames[idx])

        return self.images[idx]

    def create_timestamps_ns(self, header=True):
        fr = []
        with open(self.ts_frame_file, 'r') as f:
            fr = f.readlines()

        if header:
            fr = fr[1:]
            
        # aqualoc original timestamps are in nanoseconds (for some reason)
        # convert to seconds
        fr = [int(line.split(",")[0].strip()) for line in fr]
        fr = [str((t - fr[0])/1.0e9) for t in fr]

        count = 0
        with open(self.ts_file, 'w') as f:
            for i in range(len(fr)):
                f.write(fr[i])
                f.write("\n")
                count += 1

        fr_count = len(glob.glob(os.path.join(self.fr_path, "*."+self.img_ftype)))

        if fr_count != count:
            print("PLEASE FIX: there are " + str(count) + " timestamps but " + str(fr_count) + " actual frames.")


class advio_dataset(dataloader):
    def __init__(self, skip_preprocess=True):
        # ADVIO parameters
        self.frames_per_second = 60
        self.sampled_fps = 10
        self.img_ftype = 'png'

        # ADVIO file paths
        self.advio = os.path.join(BASE_DIR, "data/diverse_vo/advio/advio-05")
        self.main_path = os.path.join(self.advio, "iphone")
        self.gt_path = os.path.join(self.advio, "ground-truth")
        self.fr_path = os.path.join(self.main_path, "frames")

        # ADVIO files
        self.settings_file = os.path.join(self.main_path, "iphone_orb.yaml")
        self.ts_frame_file = os.path.join(self.main_path, 'frames.csv')
        self.ts_file = os.path.join(self.main_path, 'timestamps.txt')
        self.advio_gt_file = os.path.join(self.gt_path, "pose.csv")
        self.gt_file = os.path.join(self.gt_path, "pose_gt.txt")

        self.gt = []
        self.times = []
        self.image_fnames = []
        self.images = []
        self.len = 0
        self.K = np.array([])

        if not skip_preprocess:
            self.preprocess()

        # load images and times
        self.get_data()
        print("Loaded " + str(self.len) + " frames.")


    def preprocess(self):
        self.create_gt()
        self.create_timestamps()

    def create_gt(self):
        gt = []
        with open(self.advio_gt_file, 'r') as f:
            for line in f:
                gt += [line.replace(",", " ")]

        with open(self.gt_file, 'w') as f:
            for l in gt:
                f.write(l)


    def create_timestamps(self):
        fr = []
        with open(self.ts_frame_file, 'r') as f:
            fr = f.readlines()

        fr = [line.split(",")[0].strip() for line in fr]

        count = 0
        with open(self.ts_file, 'w') as f:
            rate = self.frames_per_second//self.sampled_fps
            for i in range(0, len(fr), rate):
                f.write(fr[i])
                f.write("\n")
                count += 1

        fr_count = len(glob.glob(os.path.join(self.fr_path, "*."+self.img_ftype)))

        if fr_count != count:
            print("PLEASE FIX: there are " + str(count) + " timestamps but " + str(fr_count) + " actual frames.")


class aqualoc_dataset(dataloader):
    def __init__(self, seq, skip_preprocess=True):
        # AQUALOC Parameters
        self.frames_per_second = 20
        self.img_ftype = 'png'
        self.seq = str(seq)

        # AQUALOC file paths
        self.aqualoc = os.path.join(BASE_DIR, "data/diverse_vo/aqualoc")
        self.main_path = os.path.join(self.aqualoc, "sequence_" + self.seq)
        self.gt_path = os.path.join(self.aqualoc, "gt_traj")
        self.fr_path = os.path.join(self.main_path, "undist_images")


        # AQUALOC files
        self.settings_file = os.path.join(self.aqualoc, "aqualoc_orb.yaml")
        self.ts_frame_file = os.path.join(self.main_path, 'aqua_img.csv')
        self.ts_file = os.path.join(self.main_path, 'timestamps.txt')
        self.aqualoc_gt_file = os.path.join(self.gt_path, "aqualoc_gt_traj_seq_0" + self.seq + ".txt")
        self.gt_file = os.path.join(self.gt_path, "gt_seq_0" + self.seq + ".txt")

        self.gt = []
        self.times = []
        self.image_fnames = []
        self.images = []
        self.len = 0
        self.K = np.array([])

        if not skip_preprocess:
            self.preprocess()

        # load images and times
        self.get_data()
        print("Loaded " + str(self.len) + " frames.")


    def preprocess(self):
        self.create_timestamps_ns()
        self.create_gt()


    def create_gt(self):
        gt = []
        with open(self.aqualoc_gt_file, 'r') as f:
            gt = f.readlines()

        gt = [line.split(" ") for line in gt]
        rest = [" ".join(line[1:]) for line in gt]
        tss = [int(float(ts[0])) for ts in gt]
        
        with open(self.ts_file, 'r') as f:
            self.times = f.readlines()
            self.times = [float(t) for t in self.times]

        with open(self.gt_file, 'w') as f:
            for i in range(len(gt)):
                f.write(str(self.times[tss[i]]) + " " + rest[i])



class rosario_dataset(dataloader):
    def __init__(self, skip_preprocess=True):
        # ROSARIO Parameters
        self.frames_per_second = 20
        self.img_ftype = 'png'
        self.seq = '4'
        self.len = 0

        # ROSARIO file paths
        self.rosario = os.path.join(BASE_DIR, "data/diverse_vo/rosario")
        self.main_path = os.path.join(self.rosario, "sequence0" + self.seq)
        self.fr_path = os.path.join(self.main_path, "zed/left")

        # ROSARIO files
        self.settings_file = os.path.join(self.main_path, "rosario_orb.yaml")
        self.ts_file = os.path.join(self.main_path, 'timestamps.txt')
        self.rosario_gt_file = os.path.join(self.main_path, "sequence0" + self.seq + "_gt.txt")
        self.gt_file = os.path.join(self.main_path, self.seq + "_gt.txt")

        self.gt = []
        self.times = []
        self.image_fnames = []
        self.images = []
        self.K = np.array([])

        if not skip_preprocess:
            self.preprocess()

        # load images and times
        self.get_data()
        print("Loaded " + str(self.len) + " frames.")


    def preprocess(self):
        self.create_timestamps()
        self.create_gt()


    def create_gt(self):
        gt = []
        with open(self.rosario_gt_file, 'r') as f:
            gt = f.readlines()

        gt = [line.split(" ") for line in gt]
        rest = [" ".join(line[1:]) for line in gt]
        tss = [float(ts[0]) for ts in gt]
        tss = [t - tss[0] for t in tss]
        tss = ["{:.3f}".format(x) for x in tss]
        final = [tss[i] + " " + rest[i] for i in range(len(gt))]

        with open(self.gt_file, 'w') as f:
            for line in final:
                f.write(line)


    def create_timestamps(self):
        # the times are actually the image names
        fr = sorted(glob.glob(os.path.join(self.fr_path, "*."+self.img_ftype)))
        fr = [st[st.find("left_"):] for st in fr]
        fr = [float(st.replace("left_", "").replace(".png","")) for st in fr]
        fr = [str(st-fr[0]) for st in fr]

        count = 0
        with open(self.ts_file, 'w') as f:
            for i in range(len(fr)):
                f.write(fr[i])
                f.write("\n")
                count += 1

        fr_count = len(glob.glob(os.path.join(self.fr_path, "*."+self.img_ftype)))

        if fr_count != count:
            print("PLEASE FIX: there are " + str(count) + " timestamps but " + str(fr_count) + " actual frames.")

            
class euroc_dataset(dataloader):
    def __init__(self, seq, skip_preprocess=True):
        # EUROC parameters
        self.frames_per_second = 20
        self.img_ftype = 'png'
        self.seq = "{0:0=2d}".format(seq)

        # EUROC file paths
        self.euroc = os.path.join(BASE_DIR, "data/diverse_vo/euroc")
        self.main_path = os.path.join(self.euroc, "mav_"+self.seq)
        self.gt_path = os.path.join(self.main_path, "state_groundtruth_estimate0")
        self.fr_path = os.path.join(self.main_path, "cam0/data")

        # EUROC files
        self.settings_file = os.path.join(self.euroc, "euroc_orb.yaml")
        self.ts_frame_file = os.path.join(self.main_path, 'cam0/data.csv')
        self.ts_file = os.path.join(self.main_path, 'timestamps.txt')
        self.euroc_gt_file = os.path.join(self.gt_path, "data.csv")
        self.gt_file = os.path.join(self.gt_path, "pose_gt.txt")

        self.gt = []
        self.times = []
        self.image_fnames = []
        self.images = []
        self.len = 0
        self.K = np.array([])

        if not skip_preprocess:
            self.preprocess()
        else:
            if not os.path.isfile(self.ts_file) or not os.path.isfile(self.gt_file):
                print("Could not find timestamp and/or ground truth file - re-running preprocessing")
                self.preprocess()

        # load images and times
        self.get_data()
        print("Loaded " + str(self.len) + " frames.")


    def preprocess(self):
        self.create_gt()
        self.create_timestamps_ns()

    def create_gt(self):
        gt = []
        with open(self.euroc_gt_file, 'r') as f:
            for line in f:
                gt += [line.replace(",", " ")]
            
        gt = gt[1:]
        gt = [line.split(" ")[:8] for line in gt]
        gt = [[float(line[0])] + line[1:4] + line[5:8] + [line[4]] for line in gt]
        first = gt[0][0]
        gt = [[str((line[0]-first)/1e9)] + line[1:] for line in gt]

        with open(self.gt_file, 'w') as f:
            for l in gt:
                line = " ".join(l)
                f.write(line + "\n")


