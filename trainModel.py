import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from networks.model import FullNetwork
from networks.discriminator import Discriminator
from networks.perceptualLossFeatures import vgg16
from data.NTUDataLoader import NTUDataset
from data.PanopticDataLoader import PanopticDataset
import torch.backends.cudnn as cudnn
from utils.modelIOFuncs import get_first_frame


DATASET = 'NTU'  # 'NTU' or 'Panoptic'
SERVER = 'crcv'  # 'newton' or 'crcv'

# data parameters
BATCH_SIZE = 6
CHANNELS = 3
FRAMES = 16
SKIP_LEN = 2
HEIGHT = 112
WIDTH = 112

# training parameters
NUM_EPOCHS = 1000
LR = 1e-4
STDEV = 0.1

GEN_PRETRAINED = True
DISC_PRETRAINED = True
MIN_GLOSS = 1.0
MIN_DLOSS = 1.0
pretrained_epochs = 0


def pretrained_weights_config():
    if SERVER == 'crcv':
        vgg_weights_path = '/home/yogesh/kara/REU2019/weights/vgg16-397923af.pth'
        i3d_weights_path = '/home/yogesh/kara/REU2019/weights/rgb_charades.pt'
        gen_weights_path = './weights/net_ntu_14_16_2_True_1000_0.0001_0.1.pt' if GEN_PRETRAINED else ''
        disc_weights_path = i3d_weights_path if DISC_PRETRAINED else ''

    elif SERVER == 'newton':
        vgg_weights_path = '/home/yrawat/kara/weights/vgg16-397923af.pth'
        i3d_weights_path = '/home/yrawat/kara/weights/rgb_charades.pt'
        gen_weights_path = './weights/net_ntu_14_16_2_True_1000_0.0001_0.1.pt' if GEN_PRETRAINED else ''
        disc_weights_path = i3d_weights_path if DISC_PRETRAINED else ''
    else:
        print('Server name unknown.')
        vgg_weights_path = i3d_weights_path = gen_weights_path = disc_weights_path = ''

    return vgg_weights_path, i3d_weights_path, gen_weights_path, disc_weights_path


def ntu_config():
    # NTU directory information
    if SERVER == 'crcv':
        data_root_dir = '/home/c2-2/yogesh/datasets/ntu-ard/frames-240x135/'
        if FRAMES * SKIP_LEN >= 32:
            train_split = '/home/yogesh/kara/data/train16.list'
            test_split = '/home/yogesh/kara/data/val16.list'
        else:
            train_split = '/home/yogesh/kara/data/train.list'
            test_split = '/home/yogesh/kara/data/val.list'
        param_file = '/home/yogesh/kara/data/view.params'
    elif SERVER == 'newton':
        data_root_dir = '/groups/mshah/data/ntu-ard/frames-240x135/'
        if FRAMES * SKIP_LEN >= 32:
            train_split = '/home/yrawat/kara/data/train16.list'
            test_split = '/home/yrawat/kara/data/val16.list'
        else:
            train_split = '/home/yrawat/kara/data/train.list'
            test_split = '/home/yrawat/kara/data/val.list'
        param_file = '/home/yrawat/kara/data/view.params'
    else:
        print('Server name unknown.')
        data_root_dir = train_split = test_split = param_file = ''
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    gen_weight_file = './weights/net_gen_ntu_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                             PRECROP, NUM_EPOCHS, LR, STDEV)
    disc_weight_file = './weights/net_disc_ntu_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                               PRECROP, NUM_EPOCHS, LR, STDEV)

    return data_root_dir, train_split, test_split, param_file, gen_weight_file, disc_weight_file


def panoptic_config():
    # panoptic directory information
    data_root_dir = '/home/c2-2/yogesh/datasets/panoptic/rgb_data/'
    train_split = '/home/yogesh/kara/data/panoptic/mod_train.list'
    test_split = '/home/yogesh/kara/data/panoptic/mod_test.list'
    close_cams_file = '/home/yogesh/kara/data/panoptic/closecams.list'
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    gen_weight_file = './weights/net_gen_pan_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                             PRECROP, NUM_EPOCHS, LR, STDEV)
    disc_weight_file = './weights/net_disc_pan_{}_{}_{}_{}_{}_{}_{}.pt'.format(BATCH_SIZE, FRAMES, SKIP_LEN,
                                                                               PRECROP, NUM_EPOCHS, LR, STDEV)
    return data_root_dir, train_split, test_split, close_cams_file, gen_weight_file, disc_weight_file


def print_params():
    """
    Function to print out all the custom parameter information for the experiment.
    :return: None
    """
    print('Parameters for training on {}'.format(DATASET))
    print('Batch Size: {}'.format(BATCH_SIZE))
    print('Tensor Size: ({},{},{},{})'.format(CHANNELS, FRAMES, HEIGHT, WIDTH))
    print('Skip Length: {}'.format(SKIP_LEN))
    print('Precrop: {}'.format(PRECROP))
    print('Close Views: {}'.format(CLOSE_VIEWS))
    print('Total Epochs: {}'.format(NUM_EPOCHS))
    print('Learning Rate: {}'.format(LR))


# ----------
#  Training
# ----------
def train_model(starting_epoch):
    min_recon = 1.0
    min_gloss = 1.0
    min_dloss = 1.0
    for epoch in range(starting_epoch, NUM_EPOCHS):  # opt.n_epochs):
        running_g_loss = 0.0
        running_recon_loss = 0.0
        running_vp_loss = 0.0
        running_perc_loss = 0.0
        running_d_loss = 0.0

        for batch_idx, (vp_diff, vid1, vid2) in enumerate(trainloader):
            vp_diff = vp_diff.type(torch.FloatTensor).to(device)
            vid1, vid2 = vid1.to(device), vid2.to(device)
            img1, img2 = get_first_frame(vid1), get_first_frame(vid2)
            img1, img2 = img1.to(device), img2.to(device)

            batch_size = vp_diff.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size).fill_(1.0), requires_grad=False)
            fake = Variable(FloatTensor(batch_size).fill_(0.0), requires_grad=False)

            # Configure input
            real_vids_v1, real_vids_v2 = Variable(vid1.type(FloatTensor)), Variable(vid2.type(FloatTensor))
            # labels = to_categorical(labels.numpy(), num_columns=opt.n_classes)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate a batch of images
            # gen_imgs = generator(z, label_input, code_input)
            gen_v2, vp_est = generator(vp_diff=vp_diff, vid1=vid1, img2=img2)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_v2)

            g_loss = adversarial_loss(validity, valid)
            recon_loss = criterion(gen_v2, vid2)
            vp_loss = criterion(vp_est, vp_diff)

            feat_gen = perceptual_loss(torch.reshape(gen_v2, (batch_size * FRAMES, CHANNELS, HEIGHT, WIDTH)))
            feat_gt = perceptual_loss(torch.reshape(vid2, (batch_size * FRAMES, CHANNELS, HEIGHT, WIDTH)))
            perc_losses = []
            for i in range(4):
                perc_losses.append(torch.mean(criterion(feat_gen[i], feat_gt[i])))

            total_g_loss = g_loss + recon_loss + vp_loss
            for i in range(4):
                total_g_loss += (0.25 * perc_losses[i])

            total_g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Loss for real images
            real_pred = discriminator(real_vids_v2)
            # print(real_pred.size())
            # print(valid.size())
            d_real_loss = adversarial_loss(real_pred, valid)

            # Loss for fake images
            fake_pred = discriminator(gen_v2.detach())
            # print(fake_pred.size())
            # print(fake.size())
            d_fake_loss = adversarial_loss(fake_pred, fake)

            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # --------------
            # Log Progress
            # --------------
            running_g_loss += g_loss.item()
            running_recon_loss += recon_loss.item()
            running_vp_loss += vp_loss.item()
            running_perc_loss += perc_losses[-1].item()
            running_d_loss += d_loss.item()
            if (batch_idx + 1) % 10 == 0:
                print('\tBatch {}/{} GLoss:{} ReconLoss:{} VPLoss:{} PLoss:{} DLoss:{}'.format(
                    batch_idx + 1,
                    len(trainloader),
                    "{0:.5f}".format(g_loss),
                    "{0:.5f}".format(recon_loss),
                    "{0:.5f}".format(vp_loss),
                    "{0:.5f}".format(perc_losses[-1]),
                    "{0:.5f}".format(d_loss)))

        print('Training Epoch {}/{} GLoss:{} ReconLoss:{} VPLoss:{} PLoss:{} DLoss:{}'.format(
            epoch + 1,
            NUM_EPOCHS,
            "{0:.5f}".format((running_g_loss / len(trainloader))),
            "{0:.5f}".format((running_recon_loss / len(trainloader))),
            "{0:.5f}".format((running_vp_loss / len(trainloader))),
            "{0:.5f}".format((running_perc_loss / len(trainloader))),
            "{0:.5f}".format((running_d_loss / len(trainloader)))))

        avg_recon = running_recon_loss / len(trainloader)
        avg_gloss = ((running_g_loss + running_recon_loss + running_vp_loss + running_perc_loss) / len(trainloader))
        avg_dloss = running_d_loss / len(trainloader)

        if avg_recon < min_recon or epoch == 0:
            min_recon = avg_recon
        if avg_gloss < min_gloss or epoch == 0:
            min_gloss = avg_gloss
        torch.save(generator.state_dict(), gen_weight_file[:-3] + '_{}'.format(epoch) + '.pt')
        if avg_dloss < min_dloss or epoch == 0:
            min_dloss = avg_dloss
        torch.save(discriminator.state_dict(), disc_weight_file[:-3] + '_{}'.format(epoch) + '.pt')
        print('MinRecon:{} MinGloss:{} MinDLoss:{}'.format(min_recon, min_gloss, min_dloss))


if __name__ == '__main__':
    """
    Main function to carry out the training loop.
    This function creates the generator and data loaders. Then, it trains the generator.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_ALL = True
    PRECROP = True if DATASET.lower() == 'ntu' else False
    VP_VALUE_COUNT = 1 if DATASET.lower() == 'ntu' else 3
    CLOSE_VIEWS = True if DATASET.lower() == 'panoptic' else False

    vgg_weights_path, i3d_weights_path, gen_weights_path, disc_weights_path = pretrained_weights_config()

    # generator
    generator = FullNetwork(vp_value_count=VP_VALUE_COUNT, stdev=STDEV,
                            output_shape=(BATCH_SIZE, CHANNELS, FRAMES, HEIGHT, WIDTH),
                            pretrained=True, vgg_weights_path=vgg_weights_path, i3d_weights_path=i3d_weights_path)
    if GEN_PRETRAINED:
        generator.load_state_dict(torch.load(gen_weights_path))
    generator = generator.to(device)
    # discriminator
    discriminator = Discriminator(in_channels=3, pretrained=DISC_PRETRAINED, weights_path=disc_weights_path)
    discriminator = discriminator.to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(generator)
        cudnn.benchmark = True

    # Loss functions
    criterion = nn.MSELoss()
    adversarial_loss = nn.BCELoss()
    perceptual_loss = vgg16().to(device)
    # categorical_loss = torch.nn.CrossEntropyLoss()
    # continuous_loss = torch.nn.MSELoss()

    optimizer_G = optim.Adam(generator.parameters(), lr=LR)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LR)

    if DATASET.lower() == 'ntu':
        data_root_dir, train_split, test_split, param_file, gen_weight_file, disc_weight_file = ntu_config()

        # data
        trainset = NTUDataset(root_dir=data_root_dir, data_file=train_split, param_file=param_file,
                              resize_height=HEIGHT, resize_width=WIDTH,
                              clip_len=FRAMES, skip_len=SKIP_LEN,
                              random_all=RANDOM_ALL, precrop=PRECROP)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        testset = NTUDataset(root_dir=data_root_dir, data_file=test_split, param_file=param_file,
                             resize_height=HEIGHT, resize_width=WIDTH,
                             clip_len=FRAMES, skip_len=SKIP_LEN,
                             random_all=RANDOM_ALL, precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    elif DATASET.lower() == 'panoptic':
        data_root_dir, train_split, test_split, close_cams_file, gen_weight_file, disc_weight_file = panoptic_config()

        # data
        trainset = PanopticDataset(root_dir=data_root_dir, data_file=train_split,
                                   resize_height=HEIGHT, resize_width=WIDTH,
                                   clip_len=FRAMES, skip_len=SKIP_LEN,
                                   random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                                   close_cams_file=close_cams_file, precrop=PRECROP)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        testset = PanopticDataset(root_dir=data_root_dir, data_file=test_split,
                                  resize_height=HEIGHT, resize_width=WIDTH,
                                  clip_len=FRAMES, skip_len=SKIP_LEN,
                                  random_all=RANDOM_ALL, close_views=CLOSE_VIEWS,
                                  close_cams_file=close_cams_file, precrop=PRECROP)
        testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    else:
        print('This network has only been set up to train on the NTU and panoptic datasets.')

    FloatTensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if device == 'cuda' else torch.LongTensor

    print_params()
    print(generator)
    print(discriminator)
    # if pretrained:
    #     starting_epoch = pretrained_epochs
    # else:
    #     starting_epoch = 0
    starting_epoch = 0
    train_model(starting_epoch)
