# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os.path as osp
import os
import numpy as np
import pickle
import logging

SAMPLES_WITH_MISSING_SKELETONS = ["S001C002P005R002A008","S001C002P006R001A008","S001C003P002R001A055","S001C003P002R002A012","S001C003P005R002A004","S001C003P005R002A005","S001C003P005R002A006","S001C003P006R002A008","S002C002P011R002A030","S002C003P008R001A020","S002C003P010R002A010","S002C003P011R002A007","S002C003P011R002A011","S002C003P014R002A007","S003C001P019R001A055","S003C002P002R002A055","S003C002P018R002A055","S003C003P002R001A055","S003C003P016R001A055","S003C003P018R002A024","S004C002P003R001A013","S004C002P008R001A009","S004C002P020R001A003","S004C002P020R001A004","S004C002P020R001A012","S004C002P020R001A020","S004C002P020R001A021","S004C002P020R001A036","S005C002P004R001A001","S005C002P004R001A003","S005C002P010R001A016","S005C002P010R001A017","S005C002P010R001A048","S005C002P010R001A049","S005C002P016R001A009","S005C002P016R001A010","S005C002P018R001A003","S005C002P018R001A028","S005C002P018R001A029","S005C003P016R002A009","S005C003P018R002A013","S005C003P021R002A057","S006C001P001R002A055","S006C002P007R001A005","S006C002P007R001A006","S006C002P016R001A043","S006C002P016R001A051","S006C002P016R001A052","S006C002P022R001A012","S006C002P023R001A020","S006C002P023R001A021","S006C002P023R001A022","S006C002P023R001A023","S006C002P024R001A018","S006C002P024R001A019","S006C003P001R002A013","S006C003P007R002A009","S006C003P007R002A010","S006C003P007R002A025","S006C003P016R001A060","S006C003P017R001A055","S006C003P017R002A013","S006C003P017R002A014","S006C003P017R002A015","S006C003P022R002A013","S007C001P018R002A050","S007C001P025R002A051","S007C001P028R001A050","S007C001P028R001A051","S007C001P028R001A052","S007C002P008R002A008","S007C002P015R002A055","S007C002P026R001A008","S007C002P026R001A009","S007C002P026R001A010","S007C002P026R001A011","S007C002P026R001A012","S007C002P026R001A050","S007C002P027R001A011","S007C002P027R001A013","S007C002P028R002A055","S007C003P007R001A002","S007C003P007R001A004","S007C003P019R001A060","S007C003P027R002A001","S007C003P027R002A002","S007C003P027R002A003","S007C003P027R002A004","S007C003P027R002A005","S007C003P027R002A006","S007C003P027R002A007","S007C003P027R002A008","S007C003P027R002A009","S007C003P027R002A010","S007C003P027R002A011","S007C003P027R002A012","S007C003P027R002A013","S008C002P001R001A009","S008C002P001R001A010","S008C002P001R001A014","S008C002P001R001A015","S008C002P001R001A016","S008C002P001R001A018","S008C002P001R001A019","S008C002P008R002A059","S008C002P025R001A060","S008C002P029R001A004","S008C002P031R001A005","S008C002P031R001A006","S008C002P032R001A018","S008C002P034R001A018","S008C002P034R001A019","S008C002P035R001A059","S008C002P035R002A002","S008C002P035R002A005","S008C003P007R001A009","S008C003P007R001A016","S008C003P007R001A017","S008C003P007R001A018","S008C003P007R001A019","S008C003P007R001A020","S008C003P007R001A021","S008C003P007R001A022","S008C003P007R001A023","S008C003P007R001A025","S008C003P007R001A026","S008C003P007R001A028","S008C003P007R001A029","S008C003P007R002A003","S008C003P008R002A050","S008C003P025R002A002","S008C003P025R002A011","S008C003P025R002A012","S008C003P025R002A016","S008C003P025R002A020","S008C003P025R002A022","S008C003P025R002A023","S008C003P025R002A030","S008C003P025R002A031","S008C003P025R002A032","S008C003P025R002A033","S008C003P025R002A049","S008C003P025R002A060","S008C003P031R001A001","S008C003P031R002A004","S008C003P031R002A014","S008C003P031R002A015","S008C003P031R002A016","S008C003P031R002A017","S008C003P032R002A013","S008C003P033R002A001","S008C003P033R002A011","S008C003P033R002A012","S008C003P034R002A001","S008C003P034R002A012","S008C003P034R002A022","S008C003P034R002A023","S008C003P034R002A024","S008C003P034R002A044","S008C003P034R002A045","S008C003P035R002A016","S008C003P035R002A017","S008C003P035R002A018","S008C003P035R002A019","S008C003P035R002A020","S008C003P035R002A021","S009C002P007R001A001","S009C002P007R001A003","S009C002P007R001A014","S009C002P008R001A014","S009C002P015R002A050","S009C002P016R001A002","S009C002P017R001A028","S009C002P017R001A029","S009C003P017R002A030","S009C003P025R002A054","S010C001P007R002A020","S010C002P016R002A055","S010C002P017R001A005","S010C002P017R001A018","S010C002P017R001A019","S010C002P019R001A001","S010C002P025R001A012","S010C003P007R002A043","S010C003P008R002A003","S010C003P016R001A055","S010C003P017R002A055","S011C001P002R001A008","S011C001P018R002A050","S011C002P008R002A059","S011C002P016R002A055","S011C002P017R001A020","S011C002P017R001A021","S011C002P018R002A055","S011C002P027R001A009","S011C002P027R001A010","S011C002P027R001A037","S011C003P001R001A055","S011C003P002R001A055","S011C003P008R002A012","S011C003P015R001A055","S011C003P016R001A055","S011C003P019R001A055","S011C003P025R001A055","S011C003P028R002A055","S012C001P019R001A060","S012C001P019R002A060","S012C002P015R001A055","S012C002P017R002A012","S012C002P025R001A060","S012C003P008R001A057","S012C003P015R001A055","S012C003P015R002A055","S012C003P016R001A055","S012C003P017R002A055","S012C003P018R001A055","S012C003P018R001A057","S012C003P019R002A011","S012C003P019R002A012","S012C003P025R001A055","S012C003P027R001A055","S012C003P027R002A009","S012C003P028R001A035","S012C003P028R002A055","S013C001P015R001A054","S013C001P017R002A054","S013C001P018R001A016","S013C001P028R001A040","S013C002P015R001A054","S013C002P017R002A054","S013C002P028R001A040","S013C003P008R002A059","S013C003P015R001A054","S013C003P017R002A054","S013C003P025R002A022","S013C003P027R001A055","S013C003P028R001A040","S014C001P027R002A040","S014C002P015R001A003","S014C002P019R001A029","S014C002P025R002A059","S014C002P027R002A040","S014C002P039R001A050","S014C003P007R002A059","S014C003P015R002A055","S014C003P019R002A055","S014C003P025R001A048","S014C003P027R002A040","S015C001P008R002A040","S015C001P016R001A055","S015C001P017R001A055","S015C001P017R002A055","S015C002P007R001A059","S015C002P008R001A003","S015C002P008R001A004","S015C002P008R002A040","S015C002P015R001A002","S015C002P016R001A001","S015C002P016R002A055","S015C003P008R002A007","S015C003P008R002A011","S015C003P008R002A012","S015C003P008R002A028","S015C003P008R002A040","S015C003P025R002A012","S015C003P025R002A017","S015C003P025R002A020","S015C003P025R002A021","S015C003P025R002A030","S015C003P025R002A033","S015C003P025R002A034","S015C003P025R002A036","S015C003P025R002A037","S015C003P025R002A044","S016C001P019R002A040","S016C001P025R001A011","S016C001P025R001A012","S016C001P025R001A060","S016C001P040R001A055","S016C001P040R002A055","S016C002P008R001A011","S016C002P019R002A040","S016C002P025R002A012","S016C003P008R001A011","S016C003P008R002A002","S016C003P008R002A003","S016C003P008R002A004","S016C003P008R002A006","S016C003P008R002A009","S016C003P019R002A040","S016C003P039R002A016","S017C001P016R002A031","S017C002P007R001A013","S017C002P008R001A009","S017C002P015R001A042","S017C002P016R002A031","S017C002P016R002A055","S017C003P007R002A013","S017C003P008R001A059","S017C003P016R002A031","S017C003P017R001A055","S017C003P020R001A059"]

def get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger):
    """
    Get raw bodies data from a skeleton sequence.

    Each body's data is a dict that contains the following keys:
      - joints: raw 3D joints positions. Shape: (num_frames x 25, 3)
      - colors: raw 2D color locations. Shape: (num_frames, 25, 2)
      - interval: a list which stores the frame indices of this body.
      - motion: motion amount (only for the sequence with 2 or more bodyIDs).

    Return:
      a dict for a skeleton sequence with 3 key-value pairs:
        - name: the skeleton filename.
        - data: a dict which stores raw data of each body.
        - num_frames: the number of valid frames.
    """
    ske_file = osp.join(skes_path, ske_name + '.skeleton')
    assert osp.exists(ske_file), 'Error: Skeleton file %s not found' % ske_file
    # Read all data from .skeleton file into a list (in string format)
    print('Reading data from %s' % ske_file[-29:])
    with open(ske_file, 'r') as fr:
        str_data = fr.readlines()

    num_frames = int(str_data[0].strip('\r\n'))
    frames_drop = []
    bodies_data = dict()
    valid_frames = -1  # 0-based index
    current_line = 1

    for f in range(num_frames):
        num_bodies = int(str_data[current_line].strip('\r\n'))
        current_line += 1

        if num_bodies == 0:  # no data in this frame, drop it
            frames_drop.append(f)  # 0-based index
            continue

        valid_frames += 1
        joints = np.zeros((num_bodies, 25, 3), dtype=np.float32)
        colors = np.zeros((num_bodies, 25, 2), dtype=np.float32)

        for b in range(num_bodies):
            bodyID = str_data[current_line].strip('\r\n').split()[0]
            current_line += 1
            num_joints = int(str_data[current_line].strip('\r\n'))  # 25 joints
            current_line += 1

            for j in range(num_joints):
                temp_str = str_data[current_line].strip('\r\n').split()
                joints[b, j, :] = np.array(temp_str[:3], dtype=np.float32)
                colors[b, j, :] = np.array(temp_str[5:7], dtype=np.float32)
                current_line += 1

            if bodyID not in bodies_data:  # Add a new body's data
                body_data = dict()
                body_data['joints'] = joints[b]  # ndarray: (25, 3)
                body_data['colors'] = colors[b, np.newaxis]  # ndarray: (1, 25, 2)
                body_data['interval'] = [valid_frames]  # the index of the first frame
            else:  # Update an already existed body's data
                body_data = bodies_data[bodyID]
                # Stack each body's data of each frame along the frame order
                body_data['joints'] = np.vstack((body_data['joints'], joints[b]))
                body_data['colors'] = np.vstack((body_data['colors'], colors[b, np.newaxis]))
                pre_frame_idx = body_data['interval'][-1]
                body_data['interval'].append(pre_frame_idx + 1)  # add a new frame index

            bodies_data[bodyID] = body_data  # Update bodies_data

    num_frames_drop = len(frames_drop)
    assert num_frames_drop < num_frames, \
        'Error: All frames data (%d) of %s is missing or lost' % (num_frames, ske_name)
    if num_frames_drop > 0:
        frames_drop_skes[ske_name] = np.array(frames_drop, dtype=np.int32)
        frames_drop_logger.info('{}: {} frames missed: {}\n'.format(ske_name, num_frames_drop,
                                                                    frames_drop))

    # Calculate motion (only for the sequence with 2 or more bodyIDs)
    if len(bodies_data) > 1:
        for body_data in bodies_data.values():
            body_data['motion'] = np.sum(np.var(body_data['joints'], axis=0))

    return {'name': ske_name, 'data': bodies_data, 'num_frames': num_frames - num_frames_drop}


def get_raw_skes_data():
    # # save_path = './data'
    # # skes_path = '/data/pengfei/NTU/nturgb+d_skeletons/'
    # stat_path = osp.join(save_path, 'statistics')
    #
    # skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    # save_data_pkl = osp.join(save_path, 'raw_skes_data.pkl')
    # frames_drop_pkl = osp.join(save_path, 'frames_drop_skes.pkl')
    #
    # frames_drop_logger = logging.getLogger('frames_drop')
    # frames_drop_logger.setLevel(logging.INFO)
    # frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'frames_drop.log')))
    # frames_drop_skes = dict()

    try:
        skes_name = np.loadtxt(skes_name_file, dtype=str)
    except FileNotFoundError:
        print(f"File {skes_name_file} not found, using all skeleton files in {skes_path}")
        skes_name = os.listdir(skes_path)
        skes_name = [osp.splitext(f)[0] for f in skes_name if f.endswith('.skeleton') and osp.splitext(f)[0] not in SAMPLES_WITH_MISSING_SKELETONS]
        skes_name = np.array(skes_name, dtype=str)
    
    num_files = skes_name.size
    print('Found %d available skeleton files.' % num_files)

    raw_skes_data = []
    frames_cnt = np.zeros(num_files, dtype=np.int32)

    for (idx, ske_name) in enumerate(skes_name):
        bodies_data = get_raw_bodies_data(skes_path, ske_name, frames_drop_skes, frames_drop_logger)
        raw_skes_data.append(bodies_data)
        frames_cnt[idx] = bodies_data['num_frames']
        if (idx + 1) % 1000 == 0:
            print('Processed: %.2f%% (%d / %d)' % \
                  (100.0 * (idx + 1) / num_files, idx + 1, num_files))

    with open(save_data_pkl, 'wb') as fw:
        pickle.dump(raw_skes_data, fw, pickle.HIGHEST_PROTOCOL)
    np.savetxt(osp.join(save_path, 'raw_data', 'frames_cnt.txt'), frames_cnt, fmt='%d')

    print('Saved raw bodies data into %s' % save_data_pkl)
    print('Total frames: %d' % np.sum(frames_cnt))

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    save_path = './'

    skes_path = '../nturgbd_raw/nturgb+d_skeletons/'
    stat_path = osp.join(save_path, 'statistics')
    if not osp.exists('./raw_data'):
        os.makedirs('./raw_data')

    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')
    save_data_pkl = osp.join(save_path, 'raw_data', 'raw_skes_data.pkl')
    frames_drop_pkl = osp.join(save_path, 'raw_data', 'frames_drop_skes.pkl')

    frames_drop_logger = logging.getLogger('frames_drop')
    frames_drop_logger.setLevel(logging.INFO)
    frames_drop_logger.addHandler(logging.FileHandler(osp.join(save_path, 'raw_data', 'frames_drop.log')))
    frames_drop_skes = dict()

    get_raw_skes_data()

    with open(frames_drop_pkl, 'wb') as fw:
        pickle.dump(frames_drop_skes, fw, pickle.HIGHEST_PROTOCOL)
        
