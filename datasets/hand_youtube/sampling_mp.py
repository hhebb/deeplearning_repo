import cv2
import pafy
from PIL import Image
import os
import multiprocessing as mp

'''
    youtube 영상에서 주기적으로 스냅샷을 찍어
    저장하는 멀티 프로세싱 코드.
'''

def process(video_ids):
    i, v_id = video_ids
    i += 51
    video = pafy.new(v_id)
    best = video.getbest(preftype='mp4')
    cap = cv2.VideoCapture(best.url)

    f_id = 0
    while(True):
        f_id += 1
        ret, frame = cap.read()

        if frame is not None:
            if f_id % 50 == 0:
                image = Image.fromarray(frame[..., ::-1])
                image.save(os.path.join(save_path, f'{i}_{f_id}.jpg'))
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


# save path
save_path = r'images'

# get video lists
video_ids = list()
with open(r'video_lists.txt') as f:
    data = f.read()
video_ids = {i:vid for i, vid in enumerate(data.split('\n'))}

# processing
if __name__ == '__main__':
    p = mp.Pool(12)
    p.map(process, video_ids.items())
