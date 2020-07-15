import os
import cv2
import torch
import numpy as np


def get_first_frame(vid_batch):
    """
    Function to extract the first frame from a batch of input videos.
    We extract the first frame from each of the videos input to the network so that the network can learn appearance
    conditioning from the desired views.
    :param vid_batch: (tensor) A batch of videos from which to extract only the first frame of each.
    :return: A tensor that holds all the first frames.
    """
    # get the first frame fom each vid in the batch and eliminate temporal dimension
    frames = [torch.squeeze(vid[:, :1, :, :]) for vid in vid_batch]
    # extract the batch size from the input vid_batch
    batch_size = vid_batch.size()[0]
    # create empty tensor containing batch_size images of the correct shape (matching the frames)
    imgs = torch.zeros(batch_size, *frames[0].size())
    # put all the first frames into the tensor
    for sample in range(batch_size):
        imgs[sample] = frames[sample]

    return imgs


def convert_outputs(inputs, outputs, output_dir, batch_num):
    if len(outputs) == 4:
        convert_outputs_4(inputs, outputs, output_dir, batch_num)
    elif len(outputs) == 6:
        convert_outputs_6(inputs, outputs, output_dir, batch_num)
    elif len(outputs) == 8:
        convert_outputs_8(inputs, outputs, output_dir, batch_num)
    elif len(outputs) == 10:
        convert_outputs_10(inputs, outputs, output_dir, batch_num)


def convert_outputs_4(inputs, outputs, output_dir, batch_num):
    vp_diff, vid1, vid2, img1, img2 = inputs
    gen_v1, gen_v2, rep_v1, rep_v2 = outputs

    convert_to_vid(tensor=vid1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='input')
    convert_to_vid(tensor=vid2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='input')
    convert_to_vid(tensor=gen_v1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='output')
    convert_to_vid(tensor=gen_v2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='output')
    convert_to_vid(tensor=rep_v1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='rep')
    convert_to_vid(tensor=rep_v2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='rep')


def convert_outputs_6(inputs, outputs, output_dir, batch_num):
    vp_diff, vid1, vid2, img1, img2 = inputs
    gen_v1, gen_v2, kp_v1, kp_v2, kp_v1_est, kp_v2_est = outputs

    convert_to_vid(tensor=vid1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='input')
    convert_to_vid(tensor=vid2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='input')
    convert_to_vid(tensor=gen_v1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='output')
    convert_to_vid(tensor=gen_v2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='output')
    convert_to_vid(tensor=kp_v1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='kp')
    convert_to_vid(tensor=kp_v2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='kp')
    convert_to_vid(tensor=kp_v1_est, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='kp_est')
    convert_to_vid(tensor=kp_v2_est, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='kp_est')


def convert_outputs_8(inputs, outputs, output_dir, batch_num):
    vp_diff, vid1, vid2, img1, img2 = inputs
    gen_v1, gen_v2, recon_v1, recon_v2, rep_v1, rep_v2, rep_v1_est, rep_v2_est = outputs

    convert_to_vid(tensor=vid1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='input')
    convert_to_vid(tensor=vid2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='input')
    convert_to_vid(tensor=gen_v1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='output')
    convert_to_vid(tensor=gen_v2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='output')
    convert_to_vid(tensor=recon_v1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='recon')
    convert_to_vid(tensor=recon_v2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='recon')
    convert_to_vid(tensor=rep_v1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='rep')
    convert_to_vid(tensor=rep_v2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='rep')
    convert_to_vid(tensor=rep_v1_est, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='rep_est')
    convert_to_vid(tensor=rep_v2_est, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='rep_est')


def convert_outputs_10(inputs, outputs, output_dir, batch_num):
    vp_diff, vid1, vid2, img1, img2 = inputs
    gen_v1, gen_v2, rep_v1, rep_v2, rep_v1_est, rep_v2_est, kp_v1, kp_v2, kp_v1_est, kp_v2_est = outputs

    convert_to_vid(tensor=vid1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='input')
    convert_to_vid(tensor=vid2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='input')
    convert_to_vid(tensor=gen_v1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='output')
    convert_to_vid(tensor=gen_v2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='output')

    convert_to_vid(tensor=rep_v1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='rep')
    convert_to_vid(tensor=rep_v2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='rep')
    convert_to_vid(tensor=rep_v1_est, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='rep_est')
    convert_to_vid(tensor=rep_v2_est, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='rep_est')

    convert_to_vid(tensor=kp_v1, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='kp')
    convert_to_vid(tensor=kp_v2, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='kp')
    convert_to_vid(tensor=kp_v1_est, output_dir=output_dir,
                   batch_num=batch_num, view=1, item_type='kp_est')
    convert_to_vid(tensor=kp_v2_est, output_dir=output_dir,
                   batch_num=batch_num, view=2, item_type='kp_est')


def convert_to_vid(tensor, output_dir, batch_num, view, item_type):
    """
    Function to convert a tensor to a series of .jpg video frames
    :param tensor: (tensor) The tensor to be converted.
    :param output_dir: (str) The path at which to save the video frames.
    :param view: (int) The view that the video is from: 1 or 2.
    :param batch_num: (int) The batch that the tensor is from.
    :param item_type: (str) The label for the output that is being converted; i.e. 'input', 'output', 'kp.
    :return: None
    """
    bsz, channels, frames, height, width = tensor.size()
    # loop through each video in the batch
    for i in range(bsz):
        vid = tensor[i]
        save_frames(vid, output_dir, batch_num, i + 1, view, item_type)


def save_frames(vid, output_dir, batch_num, vid_num, view, item_type):
    """
    Function to save the frames of a video to .jpgs.
    :param vid: (tensor) The video to be saved.
    :param output_dir: (str) The path at which to save the video frames.
    :param batch_num: (int) The batch that the video is from.
    :param vid_num: (int) The position of the video in the batch.
    :param view: (int) The view that the video is from: 1 or 2.
    :param item_type: (str) The label for the output that is being converted; i.e. 'input', 'output', 'kp.
    :return: None
    """
    channels, frames, height, width = vid.size()
    vid_path = make_vid_path(output_dir, batch_num, vid_num, view, item_type)

    if 'kp' in item_type:
        for kp in range(channels):
            for f in range(frames):
                hmap_name = make_frame_name(f + 1)[:-4] + make_frame_name(kp + 1)
                hmap_path = os.path.join(vid_path, hmap_name)
                hmap = vid[kp, f, :, :].cpu().numpy()
                hmap = denormalize_frame(hmap)
                try:
                    cv2.imwrite(hmap_path, hmap)
                except:
                    print('The image did not successfully save.')

                assert os.path.exists(hmap_path), 'The image does not exist.'

    if 'rep' in item_type:
        for c in range(channels, 10):
            for f in range(frames):
                hmap_name = make_frame_name(f + 1)[:-4] + make_frame_name(c + 1)
                hmap_path = os.path.join(vid_path, hmap_name)
                hmap = vid[c, f, :, :].cpu().numpy()
                hmap = denormalize_frame(hmap)
                try:
                    cv2.imwrite(hmap_path, hmap)
                except:
                    print('The image did not successfully save.')

                assert os.path.exists(hmap_path), 'The image does not exist.'

    else:
        if channels == 3:
            for i in range(frames):
                frame_name = make_frame_name(i + 1)
                frame_path = os.path.join(vid_path, frame_name)
                # extract one frame as np array
                frame = vid[:, i, :, :].squeeze().cpu().numpy()
                frame = denormalize_frame(frame)
                frame = from_tensor(frame)

                try:
                    cv2.imwrite(frame_path, frame)
                except:
                    print('The image did not successfully save.')

                assert os.path.exists(frame_path), 'The image does not exist.'


def make_vid_path(output_dir, batch_num, vid_num, view, item_type):
    """
    Function to make the path to save the video. Makes sure that the necessary paths exist.
    :param output_dir: (str) The path that holds all the video frames.
    :param batch_num: (int) The batch that the video is from.
    :param vid_num: (int) The position of the video in the batch.
    :param view: (int) The view that the video is from: 1 or 2.
    :param item_type: (str) The label for the output that is being converted; i.e. 'input', 'output', 'kp.
    :return:
    """
    batch_num, vid_num, view = str(batch_num), str(vid_num), str(view)
    batch_path = os.path.join(output_dir, batch_num)
    vid_path = os.path.join(batch_path, vid_num)
    view_path = os.path.join(vid_path, view)
    full_path = os.path.join(view_path, item_type)

    dir_list = [output_dir, batch_path, vid_path, view_path, full_path]
    for dir in dir_list:
        if not os.path.exists(dir):
            os.mkdir(dir)
    return full_path


def make_frame_name(frame_num):
    """
    Function to correctly generate the correctly formatted .jpg file name for the frame.
    :param frame_num: The frame number captured in the file.
    :return: str representing the file name.
    """
    return str(frame_num).zfill(3) + '.jpg'


def denormalize_frame(frame):
    """
    Function to denormalize the pixel values in the frame to be between 0 and 255.
    :param frame: (array-like) The frame to be denormalized.
    :return: (np array) The denormalized frame.
    """
    frame = np.array(frame).astype(np.float32)
    return np.multiply(frame, 255.0)


def from_tensor(sample):
    """
    Function to convert the sample clip from a tensor to a numpy array.
    :param sample: (tensor) The sample to convert.
    :return: a numpy array representing the sample clip.
    """
    # pytorch tensor is (channels, height, width)
    # np is (height, width, channels)
    sample = np.transpose(sample, (1, 2, 0))

    return sample


def export_vps(vp_gt, vp_est, output_dir, batch_num):
    file_path = os.path.join(output_dir, 'vps.txt')
    bsz = vp_gt.size()[0]
    assert bsz == vp_est.size()[0], 'vp_gt and vp_est are not the same size'

    with open(file_path, 'a') as f:
        f.write('Batch:{}\n'.format(batch_num))
        f.write('{}\t{}\t{}\n'.format('Sample', 'GroundTruth', 'Estimated'))
        for i in range(bsz):
            f.write('{}\t{}\t{}\n'.format(str(i + 1).ljust(6),
                                          str("{0:.5f}".format(vp_gt[i])).zfill(8),
                                          str("{0:.5f}".format(vp_est[i])).zfill(8)))
        f.write('\n')
