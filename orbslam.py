import numpy as np
import orbslam2
import cv2
import time
import math
from tqdm import tnrange, tqdm_notebook


def orbslam(imgs, timestamps, vocab, settings):
    num_images = len(imgs)

    slam = orbslam2.System(vocab, settings, orbslam2.Sensor.MONOCULAR)
    slam.set_use_viewer(False)
    slam.initialize()

    times_track = [0 for _ in range(num_images)]
    print('-----')
    print('Start processing sequence ...')
    print('Images in the sequence: {0}'.format(num_images))

    for idx in tnrange(num_images):
        image = cv2.imread(imgs[idx], cv2.IMREAD_UNCHANGED)
        tframe = timestamps[idx]

        if image is None:
            print("failed to load image at {0}".format(imgs[idx]))
            break

        t1 = time.time()
        slam.process_image_mono(image, tframe)
        t2 = time.time()

        ttrack = t2 - t1
        times_track[idx] = ttrack

        t = 0
        if idx < num_images - 1:
            t = timestamps[idx + 1] - tframe
        elif idx > 0:
            t = tframe - timestamps[idx - 1]

        if ttrack < t:
            time.sleep(t - ttrack)

    times_track = sorted(times_track)
    total_time = sum(times_track)
    print('-----')
    print('median tracking time: {0}'.format(times_track[num_images // 2]))
    print('mean tracking time: {0}'.format(total_time / num_images))

    tmp = np.expand_dims([0,0,0,1], axis=0)

    #convert pose and inverse pose into 4x4 matrices
    pose = np.array(slam.get_keyframe_poses())
    tframe = [t[0] for t in pose]
    pose = [np.concatenate((f[1:].reshape(3,4), tmp), axis=0) for f in pose]

    inverse_pose = np.array(slam.get_inverse_keyframe_poses())
    inverse_pose = [np.concatenate((f[1:].reshape(3,4), tmp), axis=0) for f in inverse_pose]
    points = [np.array(frame) for frame in slam.get_keyframe_points()]

    slam.shutdown()
    return points, pose, inverse_pose, tframe
