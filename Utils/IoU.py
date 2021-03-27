import cv2
import os


class Main:
    def measure_iou(self, img, groud_truth: list, pred: list):
        global iou, count
        self.draw_ground_truth(img, groud_truth)

        xgt = max(groud_truth[0], pred[0])
        ygt = max(groud_truth[1], pred[1])
        xpa = min(groud_truth[2], pred[2])
        ypa = min(groud_truth[3], pred[3])
        # compute the area of intersection rectangle
        interArea = max(0, xpa - xgt + 1) * max(0, ypa - ygt + 1)

        # compute the area of both the prediction and ground-truth
        groud_truthArea = (groud_truth[2] - groud_truth[0] + 1) * (groud_truth[3] - groud_truth[1] + 1)
        predArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas = interesection area
        iou += interArea / float(groud_truthArea + predArea - interArea)
        count += 1
        # return the intersection over union value

    def draw_ground_truth(self, img, groud_truth: list):
        # draw ground truth box
        tmp_img = cv2.imread(f'{img}')
        cv2.rectangle(tmp_img, (groud_truth[0], groud_truth[1]), (groud_truth[2], groud_truth[3]), (0, 255, 0), 2)
        os.remove(f'{img}')
        cv2.imwrite(f'{img}', tmp_img)

    def write_average_iou(self, img):
        tmp_img = cv2.imread(f'{img}')
        avg_iou = '{:.2f}'.format(iou / count)
        cv2.putText(tmp_img, f'Average IoU: {str(avg_iou)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    # font size, color, thickness
                    1, (0, 0, 255), 2)
        os.remove(f'{img}')
        cv2.imwrite(f'{img}', tmp_img)


iou, count = 0, 0
# x = Main()
# x.draw_ground_truth("F:\\APU\\Modules\\CP\\CP2\\Object Detection\\predicted output\\images\\20210327_165052697000.jpg",
#               [26,20,222,456])
# x.measure_iou("F:\\APU\\Modules\\CP\\CP2\\Object Detection\\predicted output\\images\\20210327_165052697000.jpg",
#               [348,286,440,480], [349,288,440,484])
# x.measure_iou("F:\\APU\\Modules\\CP\\CP2\\Object Detection\\predicted output\\images\\20210327_165052697000.jpg",
#               [404,46,746,578], [393,68,749,590])
# x.measure_iou("F:\\APU\\Modules\\CP\\CP2\\Object Detection\\predicted output\\images\\20210327_165052697000.jpg",
#               [110,534,628,798], [39,503,649,813])
# x.write_average_iou("F:\\APU\\Modules\\CP\\CP2\\Object Detection\\predicted output\\images\\20210327_165052697000.jpg")