# phase 2

import os
import time
import torch
import torch.nn as nn
from networks.model import FullNetwork
from data.NTUDataLoader import NTUDataset
from data.PanopticDataLoader import PanopticDataset
from utils.modelIOFuncs import get_first_frame, convert_to_vid, export_vps
import torch.backends.cudnn as cudnn

DATASET = 'NTU'  # 'NTU' or 'panoptic'

# data parameters
BATCH_SIZE = 14
CHANNELS = 3
FRAMES = 16
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112

STDEV = 0.1


def ntu_config():
    # NTU directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/ntu-ard/frames-240x135/'
    if FRAMES * SKIP_LEN >= 32:
        test_split = '/home/yogesh/kara/data/val16.list'
    else:
        test_split = '/home/yogesh/kara/data/val.list'
    param_file = '/home/yogesh/kara/data/view.params'
    weights_path = './weights/net_ntu_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                      PRECROP, 1000, 0.0001, STDEV)
    output_video_dir = './videos/ntu_'

    return data_root_dir, test_split, param_file, weights_path, output_video_dir


def panoptic_config():
    # panoptic directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
    test_split = '/home/yogesh/kara/data/panoptic/mod_test.list'
    close_cams_file = '/home/yogesh/kara/data/panoptic/closecams.list'
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    weights_path = './weights/net_pan_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                                PRECROP, 1000, 0.0001, STDEV)
    output_video_dir = './videos/pan_'

    return data_root_dir, test_split, close_cams_file, weights_path, output_video_dir


def test():
    """
    Function to carry out the testing/validation loop for the Full Network for a single epoch.
    :return: None
    """
    running_recon_loss = 0.0
    running_vp_loss = 0.0

    model.eval()

    for batch_idx, (vp_diff, vid1, vid2) in enumerate(testloader):
        vp_diff = vp_diff.type(torch.FloatTensor).to(device)
        vid1, vid2 = vid1.to(device), vid2.to(device)
        img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
        img1, img2 = img1.to(device), img2.to(device)

        with torch.no_grad():
            gen_v2, vp_est = model(vp_diff=vp_diff, vid1=vid1, img2=img2)

            # save videos
            convert_to_vid(tensor=vid1, output_dir=output_video_dir,
                           batch_num=batch_idx + 1, view=1, item_type='input')
            convert_to_vid(tensor=vid2, output_dir=output_video_dir,
                           batch_num=batch_idx + 1, view=2, item_type='input')
            convert_to_vid(tensor=gen_v2, output_dir=output_video_dir,
                           batch_num=batch_idx + 1, view=2, item_type='output')
            export_vps(vp_gt=vp_diff, vp_est=vp_est, output_dir=output_video_dir, batch_num=batch_idx + 1)

            # loss
            recon_loss = criterion(gen_v2, vid2)
            vp_loss = criterion(vp_est, vp_diff)
            loss = recon_loss + vp_loss

        running_recon_loss += recon_loss.item()
        running_vp_loss += vp_loss.item()
        if (batch_idx + 1) % 10 == 0:
            print('\tBatch {}/{} ReconLoss:{} VPLoss:{}'.format(
                batch_idx + 1,
                len(testloader),
                "{0:.5f}".format(recon_loss),
                "{0:.5f}".format(vp_loss)))

    print('Testing Complete ReconLoss:{} VPLoss:{}'.format(
        "{0:.5f}".format((running_recon_loss / len(testloader))),
        "{0:.5f}".format((running_vp_loss / len(testloader)))))


def test_model():
    """
    Function to test the model.
    :return: None
    """
    start_time = time.time()
    test()
    end_time = time.time()
    print('Time: {}'.format(end_time - start_time))


def print_params():
    """
    Function to print out all the custom parameter information for the experiment.
    :return: None
    """
    print('Parameters for testing on {}:'.format(DATASET))
    print('Batch size: {}'.format(BATCH_SIZE))
    print('Tensor size: ({},{},{},{})'.format(CHANNELS, FRAMES, HEIGHT, WIDTH))
    print('Skip Length: {}'.format(SKIP_LEN))
    print('Precrop: {}'.format(PRECROP))
    print('Close Views: {}'.format(CLOSE_VIEWS))


if __name__ == '__main__':
    """
    Main function to carry out the training loop. 
    This function creates the model and data loaders. Then, it trains the model.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_ALL = True
    PRECROP = True if DATASET.lower() == 'ntu' else False
    VP_VALUE_COUNT = 1 if DATASET.lower() == 'ntu' else 3
    CLOSE_VIEWS = True if DATASET.lower() == 'panoptic' else False

    if DATASET.lower() == 'ntu':
        data_root_dir, test_split, param_file, weights_path, output_video_dir = ntu_config()

        # model
        model = FullNetwork(vp_value_count=VP_VALUE_COUNT, stdev=STDEV,
                            output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH))
        model.load_state_dict(torch.load(weights_path))
        model = model.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.MSELoss()

        # data
        testset = NTUDataset(root_dir=data_root_dir, data_file=test_split, param_file=param_file,
                             resize_height=HEIGHT, resize_width=WIDTH,
                             clip_len=FRAMES, skip_len=SKIP_LEN,
                             random_all=RANDOM_ALL, precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    elif DATASET.lower() == 'panoptic':
        data_root_dir, test_split, close_cams_file, weights_path, output_video_dir = panoptic_config()

        # model
        model = FullNetwork(vp_value_count=VP_VALUE_COUNT, stdev=STDEV,
                            output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH), use_est_vp=True)
        model.load_state_dict(torch.load(weights_path))
        model = model.to(device)

        if device == 'cuda':
            net = torch.nn.DataParallel(model)
            cudnn.benchmark = True

        criterion = nn.MSELoss()

        testset = PanopticDataset(root_dir=data_root_dir, data_file=test_split,
                                  resize_height=HEIGHT, resize_width=WIDTH,
                                  clip_len=FRAMES, skip_len=SKIP_LEN,
                                  random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                                  close_cams_file=close_cams_file, precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    else:
        print('This network has only been set up to run on the NTU and panoptic datasets.')

    print_params()
    print(model)
    test_model()
