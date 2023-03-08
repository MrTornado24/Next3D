import argparse
import glob
import os
import sys
sys.path.append(os.path.abspath('.'))
import numpy as np
from util.video_preprocess.extract_3dmm import Extract3dmm
from util.video_preprocess.extract_landmark import get_landmark
import util.video_utils as video_utils
from PIL import Image
import tqdm


def calculate_3dmm_distance(s_3dmm, t_3dmm):
    """
    3dmm: B, 73
    ex_coeff = coeff_3dmm[:, 80:144]  # expression
    # tex_coeff = coeff_3dmm[:,144:224] #texture
    angles = coeff_3dmm[:, 224:227]  # euler angles for pose
    # gamma = coeff_3dmm[:,227:254] #lighting
    translation = coeff_3dmm[:, 254:257]  # translation
    crop = coeff_3dmm[:, 257:300]  # crop param

    AED: Average Expression Distance
    APD: Average Pose Distance
    :return:
    """
    assert len(s_3dmm) >= len(t_3dmm), f's_3dmm: {len(s_3dmm)}; t_3dmm: {len(t_3dmm)}.'

    s_3dmm = s_3dmm[:len(t_3dmm)]
    s_exp, s_pos = s_3dmm[:, :64], s_3dmm[:, 64:67]
    # s_exp, s_pos = s_3dmm[:, :64], s_3dmm[:, 64:70]
    t_exp, t_pos = t_3dmm[:, :64], t_3dmm[:, 64:67]
    # t_exp, t_pos = t_3dmm[:, :64], t_3dmm[:, 64:70]

    AED = np.mean(np.abs(s_exp - t_exp))
    APD = np.mean(np.abs(s_pos - t_pos))
    return AED, APD


def get_video_path_list(video_path):
    video_path_list = sorted(glob.glob(f'{video_path}/*.mp4'))
    return video_path_list


def parse_args(cmd=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_video_root', type=str, default=None)
    parser.add_argument('--target_video_root', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--extract_image', action='store_true')
    parser.add_argument('--calculate', action='store_true')
    if cmd is not None:
        args = parser.parse_args(cmd.split())
    else:
        args = parser.parse_args()
    return args


def extract_3dmm_from_videos(video_roots, args):
    extractor = Extract3dmm()
    # source_video_root = args.source_video_root
    # target_video_root = args.target_video_root
    batch_size = args.batch_size

    # video_roots = [source_video_root]
    # if target_video_root is not None:
    #     video_roots.append(target_video_root)
    for video_root in video_roots:
        source_3dmm_root = os.path.join(video_root, '3dmm')
        os.makedirs(source_3dmm_root, exist_ok=True)
        source_video_path_list = get_video_path_list(video_root)
        for video_path in tqdm.tqdm(source_video_path_list):
            frame_pil_list = video_utils.read_video(video_path, resize=256)
            lm_np = get_landmark(frame_pil_list)

            num_batch = len(frame_pil_list) // batch_size + 1

            coeff_3dmm = []
            for _i in range(num_batch):
                _frame_pil = frame_pil_list[_i * batch_size:(_i + 1) * batch_size]
                if len(_frame_pil) == 0:
                    break
                _lm_np = lm_np[_i * batch_size:(_i + 1) * batch_size]
                _3dmm_np = extractor.get_3dmm(_frame_pil, _lm_np)
                coeff_3dmm.append(_3dmm_np)
            coeff_3dmm = np.concatenate(coeff_3dmm, axis=0)

            video_name = video_path.split('/')[-1].split('.')[0]
            np.save(os.path.join(source_3dmm_root, '3dmm_' + video_name), coeff_3dmm)

        print(f'Calculate {video_root} video files 3dmm params done.')


def extract_3dmm_from_images(roots, args):
    extractor = Extract3dmm()
    # source_video_root = args.source_video_root
    # target_video_root = args.target_video_root
    batch_size = args.batch_size

    # video_roots = [source_video_root]
    # if target_video_root is not None:
    #     video_roots.append(target_video_root)
    for root in roots:
        source_3dmm_root = os.path.join(root, '3dmm')
        os.makedirs(source_3dmm_root, exist_ok=True)
        path_list = sorted(glob.glob(f'{root}/*.jpg'), key=lambda info: (int(info.split('/')[-1][:-4])))

        coeff_3dmm = []
        num_batch = len(path_list) // batch_size + 1
        for _i in tqdm.tqdm(range(num_batch)):
            _path_list = path_list[_i * batch_size:(_i + 1) * batch_size]
            if len(_path_list) == 0:
                break
            _frame_pil = [Image.open(p).resize((256, 256)) for p in _path_list]
            try:
                _lm_np = get_landmark(_frame_pil)
                _3dmm_np = extractor.get_3dmm(_frame_pil, _lm_np)
                coeff_3dmm.append(_3dmm_np)
            except:
                coeff_3dmm.append(coeff_3dmm[-1])

            if _i % 10000 == 0:
                # temp save
                _coeff_3dmm = np.concatenate(coeff_3dmm, axis=0)
                np.save(os.path.join(source_3dmm_root, '3dmm'), _coeff_3dmm)
        coeff_3dmm = np.concatenate(coeff_3dmm, axis=0)
        np.save(os.path.join(source_3dmm_root, '3dmm'), coeff_3dmm)
        print(f'Calculate {root} video files 3dmm params done.')


def cal_metrics(video_roots, args):
    source_video_root = args.source_video_root
    target_video_root = args.target_video_root
    # assert target_video_root is not None
    if target_video_root is not None:
        video_roots = [target_video_root]

    for root in video_roots:
        AED, APD = [], []
        target_video_path_list = get_video_path_list(root)
        for video_path in target_video_path_list:
            video_name = video_path.split('/')[-1].split('.')[0]
            if 'img' in video_name:
                source_video_name = video_name.split('_img_')[0]
            else:
                source_video_name = video_name
            _source_3dmm_path = os.path.join(source_video_root, '3dmm', f'3dmm_{source_video_name}.npy')
            _target_3dmm_path = os.path.join(root, '3dmm', f'3dmm_{video_name}.npy')
            _s_3dmm = np.load(_source_3dmm_path)
            _t_3dmm = np.load(_target_3dmm_path)
            _AED, _APD = calculate_3dmm_distance(_s_3dmm, _t_3dmm)
            # print(f'Video: {video_name}. 3dmm criterion value: AED: {_AED}, APD: {_APD}.')
            AED.append(_AED)
            APD.append(_APD)
        AED = np.mean(AED)
        APD = np.mean(APD)
        print(f'Target root: {root}. All videos. 3dmm criterion value: AED: {AED}, APD: {APD}.')


if __name__ == '__main__':
    # GT = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/sr_video'
    # X2Face = '/apdcephfs/share_1290939/feiiyin/TH/X2Face/UnwrapMosaic/same_id'
    # BiLayer = '/apdcephfs/share_1290939/feiiyin/TH/bilayer-model/same_id'
    # FOMM = '/apdcephfs/share_1290939/feiiyin/TH/first-order-model/same_id'
    # PIRender = '/apdcephfs/share_1290939/feiiyin/TH/PIRender/same_id'
    # FreeStyle_v40 = '/apdcephfs/share_1290939/feiiyin/TH/FreeStyle/same_id_v40'
    # FreeStyle_v42 = '/apdcephfs/share_1290939/feiiyin/TH/FreeStyle/same_id_v42'
    # FreeStyle_v40_bg = '/apdcephfs/share_1290939/feiiyin/TH/FreeStyle/same_id_v40_swapbg'
    # FreeStyle_v42_bg = '/apdcephfs/share_1290939/feiiyin/TH/FreeStyle/same_id_v42_swapbg'


    # X2Face_cs = '/apdcephfs/share_1290939/feiiyin/TH/X2Face/UnwrapMosaic/cross_id'
    # BiLayer_cs = '/apdcephfs/share_1290939/feiiyin/TH/bilayer-model/cross_id'
    # FOMM_cs = '/apdcephfs/share_1290939/feiiyin/TH/first-order-model/cross_id'
    # PIRender_cs = '/apdcephfs/share_1290939/feiiyin/TH/PIRender/cross_id'
    # FreeStyle_v40_cs = '/apdcephfs/share_1290939/feiiyin/TH/FreeStyle/cross_id_v40'
    # FreeStyle_v42_cs = '/apdcephfs/share_1290939/feiiyin/TH/FreeStyle/cross_id_v42'

    ############### new_path
    GT = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/gt/sr_video'
    X2Face = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/x2face/same_id'
    BiLayer = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/bilayer/same_id_mask'
    FOMM = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/fomm/same_id'
    PIRender = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/pirender/same_id'
    FreeStyle_v51_i = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/FreeStyle/same_id_v51_i'
    cmd = f'--source_video_root={GT} \
            --extract_image'
    #--target_video_root={FreeStyle_v42}
    video_roots = [
        GT,
        X2Face,
        BiLayer,
        FOMM,
        PIRender,
        FreeStyle_v51_i,
    ]

    X2Face_cs = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/x2face/cross_id'
    BiLayer_cs = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/bilayer/cross_id_mask'
    FOMM_cs = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/fomm/cross_id'
    PIRender_cs = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/pirender/cross_id'
    FreeStyle_v51_cs = '/apdcephfs/share_1290939/feiiyin/TH/visual_result/FreeStyle/cross_id_v51_20000_i_load'

    video_roots_cs = [
        X2Face_cs,
        BiLayer_cs,
        FOMM_cs,
        PIRender_cs,
        FreeStyle_v51_cs
    ]
    args = parse_args(cmd)
    if args.extract:
        print('3DMM mode: extract')
        extract_3dmm_from_videos(video_roots_cs, args)
    if args.extract_image:
        print('3DMM mode: extract images')
        # roots = ['/apdcephfs/share_1290939/feiiyin/data/CelebA-HQ-img/']
        # roots = ['/apdcephfs/share_1290939/feiiyin/TH/visual_result/paper/twoimages']
        roots = ['/apdcephfs/share_1290939/feiiyin/TH/giraffe/out/ffhq256_pretrained/rendering/rotation_object/source']
        # roots = ['/apdcephfs/share_1290939/feiiyin/TH/visual_result/gt/image']
        extract_3dmm_from_images(roots, args)
    if args.calculate:
        print('3DMM mode: calculate')
        cal_metrics(video_roots_cs, args)


# python comparison_experiment/3dmm/cal_3dmm_distance.py