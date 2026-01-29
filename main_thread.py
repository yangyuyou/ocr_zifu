import json
import os.path
import os
import time
import traceback
import uuid
import cv2
from config import args
from database import redisDataBase
# from multiprocessing import Process, Queue, Manager
import threading
from queue import Queue
# from onnx import yoloBoxFeaturesV2 as yoloBoxFeatures
from img_inferface import PeopleDetect
#from cos_l2 import deepSort
from video import videoConnect, delete_video_streaming, format_data, get_in_out_res, push_task_job, keep_ali
from video import push_error_video
from utils import get_current_memory_gb
import numpy as np
import sys
from count import get_state

##全局跟踪器
sys.path.append(r'/home/netmarch/wangmy/track/bytetrack')
from bytetrack.yolox.tracker.byte_tracker import BYTETracker


class BYTETrackerArgs:
    track_thresh: np.float32 = 0.25
    track_buffer: int = 30
    match_thresh: np.float32 = 0.8
    aspect_ratio_thresh: np.float32 = 3.0
    min_box_area: np.float32 = 1.0
    mot20: bool = False


tracks = {}


def simulation_data(video_frame_queue: Queue):
    """
    模拟数据灌入
    :param video_frame_queue:
    :return:
    """
    import cv2
    from glob import glob
    root = "/Users/wenyinlong/Desktop/2022/zhangxu/demo/电信/data/行人跟踪-采集/test/test/*.jpg"
    paths = glob(root)
    paths = sorted(paths, key=lambda x: int(os.path.split(x)[-1].split('.')[0].split('-')[-1]))
    uid = "test"  # uuid.uuid1().__str__()
    for ind, p in enumerate(paths[:]):
        img = cv2.imread(p)

        tmp_data = {"roi": "[[0.0, 0.0], [300.0, 0.0], [300.0, 1080.0], [0.0, 1080.0]]",
                    "task_id": uid,
                    'time_start': time.time(),
                    'index': ind,
                    "path": p
                    }
        # print(tmp_data)
        video_frame_queue.put([tmp_data, img])
        time.sleep(0.1)


def distribute_task(video_frame_queue: Queue,
                    image_dict: dict,
                    video_dict: dict,
                    task_queue_list: dict
                    ):
    """
    向算法分配任务，保证每一个摄像头都是循序逻辑处理
    :param video_dict:
    :param image_dict:
    :param video_frame_queue:
    :return:
    """
    database = redisDataBase()
    while True:
        line, frame = video_frame_queue.get()
        task_id = line["task_id"]
        ##进行任务识别
        index = line["index"]
        # image_dict["{}-{}".format(task_id, index)] = img
        ##为 taskid 寻找合适的 model_id
        model_ids = video_dict.keys()
        flag = False
        opt_model_id = None
        min_num_model_id = None
        min_num_model_num = 10000
        for model_id in model_ids:
            model_task_list = video_dict[model_id]

            if len(model_task_list) < min_num_model_num:
                min_num_model_num = len(model_task_list)
                min_num_model_id = model_id

            if task_id in model_task_list:
                flag = True
                opt_model_id = model_id

        if not flag:
            opt_model_id = min_num_model_id  ##

        if opt_model_id is not None:
            ##选择最合适的模型去计算
            if task_id not in video_dict[opt_model_id]:
                video_dict[opt_model_id] = video_dict[opt_model_id] + [task_id]
            # print("push task",
            #       "date:", time.asctime(),
            #       "model-id:", opt_model_id,
            #       "task-id:", task_id,
            #       "frame-index:", index,
            #       "video_dict:", video_dict,
            #       "line:", line,
            #       "video_frame_queue size:", video_frame_queue.qsize()
            #       )
            task_queue_list[opt_model_id].put([line, frame])
            # database.save("model-{}".format(opt_model_id), [line])


def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=1, lineType=cv2.LINE_AA)
    if label:
        w, h = cv2.getTextSize(label, 0, fontScale=2 / 3, thickness=1)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)
        cv2.putText(image,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, 2 / 3, txt_color, thickness=1, lineType=cv2.LINE_AA)


class modelConsumer:
    def __init__(self, video_dict: dict,
                 image_dict: dict,
                 warning_queue: Queue = None, uid=0, task_queue=None):
        """

        :param video_dict: 视频字典
        :param warning_queue: 告警队列
        :param gpu_id:
        """
        self.uid = uid
        self.model = PeopleDetect()
        self.run(video_dict, image_dict, warning_queue, uid, task_queue)

    def run(self, video_dict: dict,
            image_dict: dict,
            warning_queue: Queue,
            uid=None,
            task_queue: Queue = None
            ):
        """
        轮询队列进行数据推理
        :param uid:
        :param task_queue:
        :param video_dict: 模型队列
        :param image_dict: 图像队列
        :param warning_queue: 告警队列
        :return:
        """
        database = redisDataBase()
        model_id = 'model{}'.format(uid)
        video_dict[model_id] = []  ##注册，等待任务认领
        print("run : init model {}".format(uid), "video_dict:", video_dict, "init mem:", get_current_memory_gb())
        while True:
            try:
                t = time.time()
                ##获取模型id应该执行的任务，图像处理
                line, img = task_queue.get()
                # line = database.get_one_data("model-{}".format(model_id))
                # if line is None:
                #    continue

                task_id = line["task_id"]
                index = line["index"]
                fps = line['fps']
                save_interval = line['save_interval']

                if img is None:
                    continue
                roi = eval(line['roi'])
                cv2.rectangle(img,(int(roi[0][0]),int(roi[0][1])),(int(roi[1][0]),int(roi[2][1])),color=(0,0,255),thickness=2)
                
                ##上一帧数据
                # last_res = database.get("detections-{}".format(task_id))
                
                if tracks.get(task_id, "") == "":
                    # 创建一个跟踪器
                    tracks[task_id] = BYTETracker(BYTETrackerArgs(), frame_rate=fps)
                    
                st=time.time()
                boxes = self.model.run(img)
                print('模型推理时间：'+str(time.time()-st))
                print(line, img.shape, len(boxes))
                bboxes = boxes.copy()
                if boxes is not None:
                    for box in boxes:
                        box[4] = 0.95  ##把置信度调高
                st1=time.time()
                #print(bb)
                
                track_list = tracks[task_id].update(boxes[:, :5], img_info=img.shape, img_size=img.shape)
                print('跟踪推理时间：'+str(time.time()-st1))
                ##获取状态
                outputs=get_state(track_list,save_interval,fps,4,roi,(img.shape[1],img.shape[0]))
                person_in=[]
                for output in outputs:
                    if output['state']=='in':
                         person_in.append(output)
                imgroot='tmp/'+str(task_id)+'/'
                if not os.path.exists(imgroot):
                       os.makedirs(imgroot)
                
                if len(person_in)>0:
                    ##推送预警信息
                    
                    actual_time=time.time()
                    capturedTime=line['time_start']
                    diff=actual_time-capturedTime
                    print(diff)
                    #with open(imgroot+'task.txt','a',encoding='utf-8') as f:
                                #text=task_id+'-'+str(index)+'-'+str(diff)
                                #f.write(text+'\n')
                    #for p in person_in:
                        #box_label(img, p['box'], 'person', (167, 146, 11))
                    #cv2.imwrite(imgroot+str(index)+'.jpg',img)
                    
                
                
                #for tl in track_list:
                    #box_iou = self.iou(tl.tlbr, boxes[:, :4])
                    #maxindex = np.argmax(box_iou)
                    #box_label(img, boxes[maxindex], '#' + str(tl.track_id) + ' person', (167, 146, 11))
                #cv2.imwrite(imgroot+str(index)+'.jpg',img)
                current_res={'task_id':task_id}
                database.set("detections-" + str(task_id),
                              json.dumps(current_res, ensure_ascii=False))
                # for tl in track_list:
                #     box_iou = self.iou(tl.tlbr, boxes[:, :4])
                #     maxindex = np.argmax(box_iou)
                #     box_label(img, boxes[maxindex], '#' + str(tl.track_id) + ' person', (167, 146, 11))
                # imgroot = 'tmp/' + str(task_id) + '/'
                # if not os.path.exists(imgroot):
                #     os.makedirs(imgroot)
                # cv2.imwrite(imgroot + str(index) + ".jpg", img)

                #
                #     frame_diff = tl.end_frame - tl.start_frame
                #     time_diff = frame_diff*save_interval / fps
                #     if time_diff>10:
                #         ##该track_id在监控画面里停留时间超过了10s，预警
                #         imgroot=
                #         if not os.path.exists()
                #         cv2.imwrite('imgs/')

                # if last_res is not None:
                #     try:
                #         last_res = json.loads(last_res)
                #     except:
                #         last_res = {}
                #         traceback.print_exc()
                # else:
                #     last_res = {}
                # old_boxes = last_res.get("boxes")
                # if old_boxes is not None:
                #     deep_sort = deepSort(old_boxes)
                #     group_id = deep_sort.predict(boxes, score=0.7, alpha=10)
                #
                #     for i in range(len(boxes)):
                #         boxes[i]["track_id"] = group_id[i]
                #
                # else:
                #     for i in range(len(boxes)):
                #         boxes[i]["track_id"] = i + 1
                #
                # boxes = sorted(boxes, key=lambda x: x["track_id"])
                # res = format_data(task_id, boxes, roi)
                # current_res = {"task_id": task_id,
                #                "boxes": boxes,
                #                "res": res,
                #                "index": index
                #                }
                #
                # pull_res = get_in_out_res(last_res.get("res", {}), res, roi)
                # # print("*"*100)
                # # print("modelid", uid, "index", index ,"pull_res:", pull_res, "\nres:", res, "\nlast_res:", last_res.get("res", {}))
                # # print("*" * 100)
                # if not (pull_res["in"] == 0 and pull_res["out"] == 0):
                #     if last_res.get("res") is not None:
                #         ##新增taskId capturedTime
                #         pull_res["capturedTime"] = line.get("time_start", time.time())
                #         pull_res["capturedTime"] = int(pull_res["capturedTime"])
                #         pull_res["taskId"] = task_id
                #         warning_queue.put([pull_res, img, index, roi])
                #
                # t = time.time() - t
                #
                # len_res = len(current_res["res"]["data"])
                # for ind in range(len_res):
                #     current_res["res"]["data"][ind]["person_id"] = str(task_id) + '-' + str(ind + 1)
                # print(current_res)
                # database.set("detections-" + str(task_id),
                #              json.dumps(current_res, ensure_ascii=False))
                #
                # # if pull_res["in"] != 0 or pull_res["out"] != 0:
                # if not (pull_res["in"] == 0 and pull_res["out"] == 0):
                #     print("push" + "*" * 50)
                #     print("date:", time.asctime(),
                #           "time:", time.time(),
                #           "model id:", self.uid,
                #           "camera_id:", task_id,
                #           "time take:", round(t, 4),
                #           "frame index:", line["index"],
                #           "last index:", last_res.get("index"),
                #           "pull_res:", pull_res,
                #           "res:", res,
                #           "last_res:", last_res.get("res"),
                #           "mem:", get_current_memory_gb()
                #           )
                #     print("end" + "*" * 50)

            except:
                traceback.print_exc()

    def iou(self, box: np.ndarray, boxes: np.ndarray):
        xy_max = np.minimum(boxes[:, 2:], box[2:])
        xy_min = np.maximum(boxes[:, :2], box[:2])
        inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
        inter = inter[:, 0] * inter[:, 1]

        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_box = (box[2] - box[0]) * (box[3] - box[1])

        return inter / (area_box + area_boxes - inter)


class Job:
    def __init__(self, model_num=args.model_num, video_num=args.video_num, warning_num=10):
        """
        任务启动
        """
        # ManagerObj = Manager()
        frame_queue = Queue(maxsize=args.video_num * 10)  ##抽帧队列最大队列长度
        warning_queue = Queue(maxsize=args.video_num * 30)  ##告警队列
        erro_queue = Queue(maxsize=args.video_num * 2)  ##异常告警队列
        video_dict = {}
        image_dict = {}
        # video_task_queue = Manager().dict()
        # print(video_dict)
        # print(image_dict)
        ##模型消费队列,负责从缓存区(抽帧队列,抽帧图片和视频信息)获取数据
        video_task_queue = {}
        for ind in range(model_num):
            # 每个模型会有一个进程
            video_task_queue['model{}'.format(ind)] = Queue(maxsize=10)

        self.process_list = []

        # ##初始化视频连接
        for ind in range(video_num):
            ##
            p = threading.Thread(target=videoConnect,
                                 args=(args.video_num * 10, frame_queue, args.frame_interval, 300, ind, erro_queue,
                                       image_dict))
            self.process_list.append(p)

        ##分配任务
        #p = threading.Thread(target=distribute_task,
                             #args=(frame_queue, image_dict, video_dict, video_task_queue))
        #self.process_list.append(p)

        ##消费模型，图像处理
        for ind in range(model_num):
            print(ind, video_task_queue['model{}'.format(ind)])
            p = threading.Thread(target=modelConsumer, args=(
            video_dict, image_dict, warning_queue, ind, frame_queue))
            self.process_list.append(p)

        # ##推送预警信息
        # for i in range(warning_num):
        #     p = threading.Thread(target=push_task_job, args=(warning_queue, i))
        #     self.process_list.append(p)

        ##删除断开连接的摄像头
        p = threading.Thread(target=delete_video_streaming, args=(300, erro_queue))
        self.process_list.append(p)

        # ##心跳检测
        # p = Process(target=keep_ali, args=(1,))
        # self.process_list.append(p)

        # ##错误流
        # p = threading.Thread(target=push_error_video, args=(erro_queue,))
        # self.process_list.append(p)

        # p = threading.Thread(target=push_error_video, args=(erro_queue,))
        # self.process_list.append(p)

        # if args.test:
        # p = threading.Thread(target=simulation_data, args=(frame_queue,))
        # self.process_list.append(p)

        print(len(self.process_list))

        for pc in self.process_list:
            pc.start()

        for pc in self.process_list:
            pc.join()

        index = 0
        for t in self.process_list:
            if t.is_alive():
                print(str(index) + ' is still alive')
            else:
                print(str(index) + ' is not alive')
            index += 1

        ##初始化视频连接
        # for ind in range(video_num):
        #     ##
        #     p = Process(target=videoConnect,
        #                 args=(args.video_num * 10, frame_queue, args.frame_interval, 300, ind, erro_queue, image_dict))
        #     self.process_list.append(p)
        #
        # ##分配任务
        # p = Process(target=distribute_task,
        #             args=(frame_queue, image_dict, video_dict, video_task_queue))
        # self.process_list.append(p)
        #
        # ##消费模型，图像处理
        # for ind in range(model_num):
        #     print(ind, video_task_queue['model{}'.format(ind)])
        #     p = Process(target=modelConsumer,
        #                 args=(video_dict, image_dict, warning_queue, ind, video_task_queue['model{}'.format(ind)]))
        #     self.process_list.append(p)
        #
        # ##推送预警信息
        # for i in range(warning_num):
        #     p = Process(target=push_task_job, args=(warning_queue, i))
        #     self.process_list.append(p)
        #
        # ##删除断开连接的摄像头
        # p = Process(target=delete_video_streaming, args=(300, erro_queue))
        # self.process_list.append(p)
        #
        # # ##心跳检测
        # # p = Process(target=keep_ali, args=(1,))
        # # self.process_list.append(p)
        #
        # # ##错误流
        # p = Process(target=push_error_video, args=(erro_queue,))
        # self.process_list.append(p)
        #
        # p = Process(target=push_error_video, args=(erro_queue,))
        # self.process_list.append(p)
        #
        # # if args.test:
        # #     p = Process(target=simulation_data, args=(frame_queue,))
        # #     self.process_list.append(p)
        #
        # for pc in self.process_list:
        #     pc.start()
        #
        # for pc in self.process_list:
        #     pc.join()


if __name__ == '__main__':
    Job()
