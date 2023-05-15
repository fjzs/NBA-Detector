import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.io import read_image
import cv2
import numpy as np

CLASS_COLORS = {
    0 : "red",
    1 : "green",
    2 : "blue"
}

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1

def drawpoly(img,pts,color,thickness=1,style='dotted',):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness,style)

def drawrect(img,pt1,pt2,color,thickness=1,style='dotted'):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness,style)

def visualize_one_image(image: torch.Tensor, prediction: dict(torch.Tensor), ground_truth: dict(torch.Tensor)) -> np.ndarray :
    """
    
    """

    pred_boxes = prediction['boxes']
    gt_boxes = ground_truth['boxes']

    pred_labels = prediction['labels']
    gt_labels = ground_truth['labels']

    color_codes = torch.tensor([CLASS_COLORS[label.item()] for label in [0,1,2]])

    vis_image = draw_bounding_boxes(image, pred_boxes, pred_labels, colors = color_codes)

    vis_image = vis_image.numpy()

    for j in range(gt_boxes.shape[0]):
        box = gt_boxes[j].numpy()
        label = gt_labels[j].item()
        
        
        x_min, y_min, x_max, y_max = box
        
        color = class_colors[label]
        cv2.rectangle(vis_image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), 'white', 2)
        drawrect(vis_image,(int(x_min), int(y_min)), (int(x_max), int(y_max)), CLASS_COLORS[label], 2, 'dotted')

    return vis_image


if __name__ == '__main__':


    pred_boxes = torch.tensor([[50, 50, 100, 200], [210, 150, 350, 430], [220, 190, 320, 330]], dtype=torch.float)
    pred_labels = torch.tensor([0,1,2])
    prediction = {}
    prediction['boxes'] = pred_boxes
    prediction['labels'] = pred_labels

    gt_boxes = torch.tensor([[20, 50, 200, 200], [210, 210, 360, 480], [220, 220, 300, 310]], dtype=torch.float)
    gt_labels = torch.tensor([0,1,2])
    ground_truth = {}
    ground_truth['boxes'] = gt_boxes
    ground_truth['labels'] = gt_labels

    img = visualize_one_image(read_image('dog.jpg'), prediction, ground_truth)

    cv2.imwrite('test.jpg',img)

# result = draw_bounding_boxes(dog1_int, boxes, colors=colors, width=5)
# show(result)
