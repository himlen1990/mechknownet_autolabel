import argparse
from datetime import datetime
import os.path as osp

import cv2
import torch
from eos import make_fancy_output_dir

from srhandnet import SRHandNet
from srhandnet.visualization import visualize_hand_keypoints
from srhandnet.visualization import visualize_hand_rects
from srhandnet.visualization import visualize_text


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--device', type=int, default=-1,
                        help='GPU device type. If -1, use cpu.')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--out', type=str, default='result')
    args = parser.parse_args()

    if args.device >= 0 and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    model = SRHandNet(threshold=0.5)
    model.eval()
    model.to(device)

    capture = cv2.VideoCapture(0)

    if args.save:
        idx = 0
        output_dir = make_fancy_output_dir(args.out, args=args)

    while True:
        ret, org_frame = capture.read()
        if ret is False:
            break
        frame = org_frame.copy()

        start = datetime.now()
        keypoints, handrect = model.pyramid_inference(
            frame, return_bbox=True)

        end = datetime.now()
        fps = 1.0 / (end - start).total_seconds()
        visualize_hand_rects(frame, handrect)
        visualize_hand_keypoints(frame, keypoints)
        frame = visualize_text(frame, 'FPS: %.1f' % fps)

        if args.save:
            for _, rect in handrect:
                y1, x1, y2, x2 = map(int, rect)
                cv2.imwrite(osp.join(output_dir, '{0:08}.jpg'.format(idx)),
                            org_frame[y1:y2, x1:x2])
                idx += 1

        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
