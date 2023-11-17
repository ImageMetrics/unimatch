import argparse
import collections
import os

import numpy as np
import torch
import torch.nn.functional as F

from unimatch.unimatch import UniMatch

NamedRange = collections.namedtuple('NamedRange', ['name', 'range'])
_COMPARISON_REGIONS = {
    '1-1 Scale': [
        NamedRange(name='Brows: non-specific', range=range(192, 245)),
        NamedRange(name='Brows: blink', range=range(335, 364)),
        NamedRange(name='Nose: lighting', range=range(341, 502)),
        NamedRange(name='Brows: lighting', range=range(770, 898)),
        NamedRange(name='Brows: lighting', range=range(934, 976)),
    ],
    'FaceCapture_Catt_Act6.1Scene1': [
        NamedRange(name='Nose: non-specific', range=range(860, 1200)),
        NamedRange(name='Brows: lighting', range=range(4429, 4516)),
        NamedRange(name='Brows: non-specific', range=range(2130, 2325)),
    ],
    'FaceCapture_Catt_Act7Scene1': [
        NamedRange(name='Brows: expression', range=range(435, 890)),
        NamedRange(name='Brows: blink', range=range(0, 200)),
        NamedRange(name='Nose: non-specific', range=range(665, 811)),
        NamedRange(name='Nose: blink', range=range(3505, 3685)),
    ],
    'FaceCapture_Eddy_Act6.1Scene1': [
        NamedRange(name='Brows: expression', range=range(1000, 1310)),
        NamedRange(name='Brows: blink', range=range(194, 229)),
        NamedRange(name='Nose: expression', range=range(1020, 1262)),
        NamedRange(name='Nose: expression', range=range(2774, 3010)),
        NamedRange(name='Brows: non-specific', range=range(3640, 3870)),
    ],
    'FaceCapture_Eddy_Act10.1Scene1': [
        NamedRange(name='Brows: blink', range=range(0, 300)),
        NamedRange(name='Brows: expression', range=range(850, 1100)),
    ],
    'FaceCapture_Eddy_Act10.5Scene1': [
        NamedRange(name='Nose: non-specific', range=range(560, 700)),
        NamedRange(name='Brows: non-specific', range=range(850, 1100)),
    ],
    'Fin_HiNIS_Node_07': [
        NamedRange(name='Brows: expression', range=range(0, 207)),
        NamedRange(name='Brows: expression', range=range(300, 460)),
    ],
    'Lucas_AFD_Demo_TP_HMC_tk06': [
        NamedRange(name='Brows: expression', range=range(0, 309)),
        NamedRange(name='Brows: blink', range=range(349, 381)),
        NamedRange(name='Nose: expression', range=range(223, 345)),
        NamedRange(name='Nose: expression', range=range(450, 610)),
        NamedRange(name='Brows: expression', range=range(940, 1070)),
    ],
    'RichardCotton_ROM_Line_Neutral': [
        NamedRange(name='Brows: expression', range=range(185, 454)),
        NamedRange(name='Brows: blink', range=range(785, 849)),
        NamedRange(name='Nose: expression', range=range(444, 610)),
    ],
    'RichardCotton_TestLine_04': [
        NamedRange(name='Nose: non-specific', range=range(175, 400)),
    ],
    'RichardCotton_TestLine_06': [
        NamedRange(name='Brows: expression', range=range(500, 800)),
        NamedRange(name='Nose: expression', range=range(0, 200)),
    ],
    'RichardCotton_TestLine_09': [
        NamedRange(name='Brows: expression', range=range(400, 535)),
        NamedRange(name='Nose: expression', range=range(150, 250)),
        NamedRange(name='Brows: blink', range=range(580, 620)),
    ],
    'ROM_CarloMestroni_20221128_055_01_Top': [
        NamedRange(name='Brows: expression', range=range(2231, 2365)),
        NamedRange(name='Brows: blink', range=range(18, 205)),
        NamedRange(name='Nose: blink', range=range(320, 430)),
        NamedRange(name='Nose: expression', range=range(2385, 2832)),
    ],
    'song_BossChick__v1_t5_STa_01_F_STa': [
        NamedRange(name='Brows: non-specific', range=range(0, 107)),
        NamedRange(name='Brows: expression', range=range(1093, 1313)),
        NamedRange(name='Nose: non-specific', range=range(2197, 2409)),
    ],
    'song_IcyGRL__v1_t10_STa_01_F_STa': [
        NamedRange(name='Nose: blink', range=range(105, 120)),
        NamedRange(name='Brows: expression', range=range(1210, 1350)),
        NamedRange(name='Brows: expression', range=range(1830, 1870)),
        NamedRange(name='Brows: blink', range=range(0, 200)),
    ],
    'video_2023-06-05_17-26-06': [
        NamedRange(name='Brows: expression', range=range(0, 127)),
        NamedRange(name='Brows: expression', range=range(335, 420)),
        NamedRange(name='Brows: blink', range=range(564, 610)),
    ],
}
ROOT_DIR = r'z:\LocalWorkingRoot\SLPT'
OUTPUT_DIR = r'z:\LocalWorkingRoot\unimatch'


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=326, type=int)

    # model: learnable parameters
    parser.add_argument('--task', default='flow', choices=['flow', 'stereo', 'depth'], type=str)
    parser.add_argument('--num_scales', default=1, type=int,
                        help='feature scales: 1/8 or 1/8 + 1/4')
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--upsample_factor', default=8, type=int)
    parser.add_argument('--num_head', default=1, type=int)
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_layers', default=6, type=int)
    parser.add_argument('--reg_refine', action='store_true',
                        help='optional task-specific local regression refinement')
    parser.add_argument('--padding_factor', default=16, type=int,
                        help='the input should be divisible by padding_factor, otherwise do padding or resizing')

    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=str,
                        help='resume from pretrained model or resume from unexpectedly terminated training')

    # model: parameter-free
    parser.add_argument('--attn_type', default='swin', type=str,
                        help='attention function')
    parser.add_argument('--attn_splits_list', default=[2], type=int, nargs='+',
                        help='number of splits in attention')
    parser.add_argument('--corr_radius_list', default=[-1], type=int, nargs='+',
                        help='correlation radius for matching, -1 indicates global matching')
    parser.add_argument('--prop_radius_list', default=[-1], type=int, nargs='+',
                        help='self-attention radius for propagation, -1 indicates global attention')
    parser.add_argument('--num_reg_refine', default=1, type=int,
                        help='number of additional local regression refinement')

    parser.add_argument('--pred_bidir_flow', action='store_true',
                        help='predict bidirectional flow')

    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--fixed_inference_size', default=None, type=int, nargs='+',
                        help='can specify the inference size for the input to the network')

    return parser


def get_test_data(track_only_regions=True):
    # test data file
    import glob
    test_files = glob.glob(os.path.join(ROOT_DIR, r'Results\LDSDK\*.npz'))

    all_landmarks = []
    all_image_files = []
    for test_data_file in test_files:
        npz_file = np.load(test_data_file)
        if track_only_regions:
            video_name = os.path.basename(test_data_file)[:-11]
            if video_name not in _COMPARISON_REGIONS.keys():
                raise RuntimeError(f'Missing video regions: {video_name}')
            image_files = []
            landmarks = []
            for region in _COMPARISON_REGIONS[video_name]:
                image_files.extend(
                    npz_file['image_files'][region.range]
                )
                landmarks.append(
                    npz_file['landmarks'][region.range, :, :]
                )
            landmarks = np.concatenate(landmarks, axis=0)
        else:
            image_files = npz_file['image_files']
            landmarks = npz_file['landmarks']
        all_landmarks.append(landmarks)
        all_image_files.append(image_files)

    return all_image_files, all_landmarks


def motion_from_flow(flow, landmarks):
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator((range(flow.shape[0]), range(flow.shape[1])), flow)
    uv = interp(landmarks[:, [1, 0]])
    landmarks_out = landmarks.copy()
    # image is transposed
    landmarks_out[:, 0] += uv[:, 1]
    landmarks_out[:, 1] += uv[:, 0]
    return landmarks_out


def adjust_flow_with_tracking(flow_landmarks, landmarks):
    # distance between eyes
    norm_dist = np.linalg.norm(landmarks[28, :] - landmarks[26, :])

    # sigmas
    sigma_x = np.zeros(85) + 1e-5
    sigma_x[0:16] = 0.05  # BROWS
    sigma_x[67:75] = 0.05  # NOSE
    sigma_y = np.zeros(85) + 1e-5
    sigma_y[0:16] = 0.025  # BROWS
    sigma_y[67:75] = 0.05  # NOSE

    d = (flow_landmarks - landmarks) / norm_dist
    w_x = np.exp(-0.5 * np.square(d[:, 0] / sigma_x))
    w_y = np.exp(-0.5 * np.square(d[:, 1] / sigma_y))

    out_landmarks = flow_landmarks.copy()
    out_landmarks[:, 0] = out_landmarks[:, 0] * w_x + landmarks[:, 0] * (1 - w_x)
    out_landmarks[:, 1] = out_landmarks[:, 1] * w_y + landmarks[:, 1] * (1 - w_y)
    return out_landmarks


@torch.no_grad()
def compute_flow(model, image1, image2,
                 retain_inference_size=False,
                 fixed_inference_size=None,
                 padding_factor=32,
                 attn_type='swin',
                 attn_splits_list=None,
                 corr_radius_list=None,
                 prop_radius_list=None,
                 num_reg_refine=1,
                 pred_bidir_flow=False,
                 ):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image1 = np.array(image1).astype(np.uint8)
    image2 = np.array(image2).astype(np.uint8)

    if len(image1.shape) == 2:  # gray image
        image1 = np.tile(image1[..., None], (1, 1, 3))
        image2 = np.tile(image2[..., None], (1, 1, 3))
    else:
        image1 = image1[..., :3]
        image2 = image2[..., :3]

    image1 = torch.from_numpy(image1).permute(2, 0, 1).float().unsqueeze(0).to(device)
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float().unsqueeze(0).to(device)

    # the model is trained with size: width > height
    transpose_img = False
    if image1.size(-2) > image1.size(-1):
        image1 = torch.transpose(image1, -2, -1)
        image2 = torch.transpose(image2, -2, -1)
        transpose_img = True

    nearest_size = [int(np.ceil(image1.size(-2) / padding_factor)) * padding_factor,
                    int(np.ceil(image1.size(-1) / padding_factor)) * padding_factor]

    # resize to nearest size or specified size
    inference_size = nearest_size if fixed_inference_size is None else fixed_inference_size

    ori_size = image1.shape[-2:]

    # resize before inference
    if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
        image1 = F.interpolate(image1, size=inference_size, mode='bilinear',
                               align_corners=True)
        image2 = F.interpolate(image2, size=inference_size, mode='bilinear',
                               align_corners=True)

    results_dict = model(image1, image2,
                         attn_type=attn_type,
                         attn_splits_list=attn_splits_list,
                         corr_radius_list=corr_radius_list,
                         prop_radius_list=prop_radius_list,
                         num_reg_refine=num_reg_refine,
                         task='flow',
                         pred_bidir_flow=pred_bidir_flow,
                         )

    flow_pr = results_dict['flow_preds'][-1]  # [B, 2, H, W]

    # resize back if required
    if not retain_inference_size:
        if inference_size[0] != ori_size[0] or inference_size[1] != ori_size[1]:
            flow_pr = F.interpolate(flow_pr, size=ori_size, mode='bilinear',
                                    align_corners=True)
            flow_pr[:, 0] = flow_pr[:, 0] * ori_size[-1] / inference_size[-1]
            flow_pr[:, 1] = flow_pr[:, 1] * ori_size[-2] / inference_size[-2]

    if transpose_img:
        flow_pr = torch.transpose(flow_pr, -2, -1)

    flow = flow_pr[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2]

    return flow


def compute_flow_video(model, image_files,
                       retain_inference_size=False,
                       redo_flow=False,
                       display=False,
                       **kwargs
                       ):
    if display:
        import matplotlib.pyplot as plt
        fig_flow, ax_flow = plt.subplots(nrows=1, ncols=2)
        cb1 = cb2 = None

    from utils import frame_utils
    import tqdm
    os.makedirs(os.path.join(OUTPUT_DIR, 'flow'), exist_ok=True)
    video_name = os.path.basename(os.path.dirname(image_files[0]))
    for test_id in tqdm.tqdm(range(0, len(image_files) - 1), desc=video_name):
        basename = os.path.basename(image_files[test_id + 1])
        output_file_name = os.path.join(OUTPUT_DIR, 'flow', f'{basename}.flow.npz')
        if redo_flow or not os.path.exists(output_file_name):
            image1 = frame_utils.read_gen(image_files[test_id])
            image2 = frame_utils.read_gen(image_files[test_id + 1])

            flow = compute_flow(model, image1, image2,
                                retain_inference_size=retain_inference_size,
                                **kwargs
                                )
            np.savez(output_file_name, flow=flow)

            if display:
                if cb1 is not None:
                    cb1.remove()
                    cb2.remove()
                ax_flow[0].cla()
                im1 = ax_flow[0].imshow(flow[:, :, 0])
                ax_flow[1].cla()
                im2 = ax_flow[1].imshow(flow[:, :, 1])
                cb1 = fig_flow.colorbar(im1, ax=ax_flow[0])
                cb2 = fig_flow.colorbar(im2, ax=ax_flow[1])
                fig_flow.show()
                fig_flow.canvas.draw()
                plt.pause(0.001)


def optical_flow_compute_all(args,
                             redo_flow=False,
                             display=False,
                             retain_inference_size=False,
                             ):
    all_image_files, all_landmarks = get_test_data()

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True

    args.distributed = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model
    model = UniMatch(feature_channels=args.feature_channels,
                     num_scales=args.num_scales,
                     upsample_factor=args.upsample_factor,
                     num_head=args.num_head,
                     ffn_dim_expansion=args.ffn_dim_expansion,
                     num_transformer_layers=args.num_transformer_layers,
                     reg_refine=args.reg_refine,
                     task=args.task).to(device)

    print('Load checkpoint: %s' % args.resume)
    loc = 'cuda:{}'.format(args.local_rank) if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.resume, map_location=loc)

    model.load_state_dict(checkpoint['model'], strict=False)

    model.eval()

    for image_files in all_image_files:
        compute_flow_video(model, image_files,
                           redo_flow=redo_flow,
                           display=display,
                           retain_inference_size=retain_inference_size,
                           padding_factor=args.padding_factor,
                           attn_type=args.attn_type,
                           attn_splits_list=args.attn_splits_list,
                           corr_radius_list=args.corr_radius_list,
                           prop_radius_list=args.prop_radius_list,
                           pred_bidir_flow=args.pred_bidir_flow,
                           num_reg_refine=args.num_reg_refine,
                           fixed_inference_size=args.fixed_inference_size,
                           )


def flow_smooth_video(image_files, landmarks,
                      display=False):
    if display:
        import matplotlib.pyplot as plt
        fig_flow, ax_flow = plt.subplots(nrows=1, ncols=2)
        cb1 = cb2 = None
        fig_im, ax_im = plt.subplots(nrows=1, ncols=2)

    from utils import frame_utils
    import tqdm

    # image size
    from PIL import Image
    im = Image.open(image_files[0])
    width, height = im.size

    os.makedirs(os.path.join(OUTPUT_DIR, 'smoothed'), exist_ok=True)
    video_name = os.path.basename(os.path.dirname(image_files[0]))
    landmarks_flow = landmarks[0, :, :]
    smoothed_landmarks = [landmarks_flow, ]
    for test_id in tqdm.tqdm(range(0, len(image_files) - 1), desc=video_name):
        basename = os.path.basename(image_files[test_id + 1])
        flow_file_name = os.path.join(OUTPUT_DIR, 'flow', f'{basename}.flow.npz')
        if not os.path.exists(flow_file_name):
            raise RuntimeError(f'Missing flow file: {flow_file_name}')
        npz_file = np.load(flow_file_name)
        flow = npz_file['flow']

        # scale to flow size and back
        scale_y = height / flow.shape[0]
        scale_x = width / flow.shape[1]
        landmarks_scaled = landmarks_flow / [scale_x, scale_y]
        landmarks_flow_updated = motion_from_flow(flow, landmarks_scaled)
        landmarks_flow_updated = landmarks_flow_updated * [scale_x, scale_y]

        # ensure consistency with tracking
        landmarks_flow = adjust_flow_with_tracking(landmarks_flow_updated, landmarks[test_id + 1, :, :])

        smoothed_landmarks.append(landmarks_flow)

        if display:
            image1 = frame_utils.read_gen(image_files[test_id])
            image2 = frame_utils.read_gen(image_files[test_id + 1])

            image1 = np.array(image1).astype(np.uint8)
            image2 = np.array(image2).astype(np.uint8)

            if cb1 is not None:
                cb1.remove()
                cb2.remove()
            ax_flow[0].cla()
            im1 = ax_flow[0].imshow(flow[:, :, 0])
            ax_flow[1].cla()
            im2 = ax_flow[1].imshow(flow[:, :, 1])
            cb1 = fig_flow.colorbar(im1, ax=ax_flow[0])
            cb2 = fig_flow.colorbar(im2, ax=ax_flow[1])
            fig_flow.show()
            fig_flow.canvas.draw()

            ax_im[0].cla()
            ax_im[0].imshow(image1)
            ax_im[0].scatter(landmarks[test_id, :, 0], landmarks[test_id, :, 1])
            ax_im[1].cla()
            ax_im[1].imshow(image2)
            ax_im[1].scatter(landmarks[test_id, :, 0], landmarks[test_id, :, 1], marker='.')
            ax_im[1].scatter(landmarks[test_id + 1, :, 0], landmarks[test_id + 1, :, 1], marker='+')
            ax_im[1].scatter(landmarks_flow_updated[:, 0], landmarks_flow_updated[:, 1], marker='.')
            ax_im[1].scatter(landmarks_flow[:, 0], landmarks_flow[:, 1], marker='+')
            fig_im.show()
            fig_im.canvas.draw()
            plt.pause(0.001)
            pass

    smoothed_landmarks = np.array(smoothed_landmarks)
    output_file_name = os.path.join(OUTPUT_DIR, 'smoothed', f'{video_name}.smoothed.npz')
    np.savez(output_file_name,
             landmarks=landmarks,
             smoothed_landmarks=smoothed_landmarks,
             image_files=image_files)
    return smoothed_landmarks


def optical_flow_smooth_all(display=False):
    all_image_files, all_landmarks = get_test_data()

    for image_files, landmarks in zip(all_image_files, all_landmarks):
        flow_smooth_video(image_files, landmarks,
                          display=display,
                          )


def main(args):
    # optical_flow_compute_all(args,
    #                          redo_flow=False,
    #                          display=False,
    #                          retain_inference_size=True)

    optical_flow_smooth_all(display=False, )


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)
