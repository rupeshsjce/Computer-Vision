import cv2
import os
import argparse
import numpy as np
import xml.etree.ElementTree as ET


def get_intersection_area(box1, box2):
    """
    Calculates the intersection area of two bounding boxes where (x1,y1) indicates the top left corner and (x2,y2)
    indicates the bottom right corner
    :param box1: List of coordinates(x1,y1,x2,y2) of box1
    :param box2: List of coordinates(x1,y1,x2,y2) of box2
    :return: float: area of intersection of the two boxes
    """
    x1 = max(box1[0], box2[0])
    x2 = min(box1[2], box2[2])
    y1 = max(box1[1], box2[1])
    y2 = min(box1[3], box2[3])
    # Check for the condition if there is no overlap between the bounding boxes (either height or width
    # of intersection box are negative)
    if (x2 - x1 < 0) or (y2 - y1 < 0):
        return 0.0
    else:
        return (x2 - x1 + 1) * (y2 - y1 + 1)


def bb_intersection_over_union(boxA, boxB):
    """
    @brief compute the intersaction over union (IoU) of two given bounding boxes
    @param boxA numpy array (x_min, y_min, x_max, y_max) # left-top and right-bottom points of the box.
    @param boxB numpy array (x_min, y_min, x_max, y_max)
    """
    ##################################################
    # TODO: Implement the IoU function               #
    ##################################################
    xA_min, yA_min, xA_max, yA_max = boxA
    area_boxA = abs(xA_min - xA_max) * abs(yA_min - yA_max)

    xB_min, yB_min, xB_max, yB_max = boxB
    area_boxB = abs(xB_min - xB_max) * abs(yB_min - yB_max)

    # Calculate intersection area
    i_LT = [max(xA_min, xB_min), min(yA_max, yB_max)] # intersection left-top point
    i_RB = [min(xA_max, xB_max), max(yA_min, yB_min)] # intersection right-bottom point

    areaI = 0
    if i_RB[0] - i_LT[0] > 0 and i_LT[1] - i_RB[1] > 0: # i_RB[0] > i_LT[0] and i_LT[1] > i_RB[1]
        areaI = (i_RB[0] - i_LT[0]) * (i_LT[1] - i_RB[1])
    
    iou = areaI / (area_boxA + area_boxB - areaI)
    '''
    area_boxA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_boxB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    intersection_area = get_intersection_area(boxA, boxB)
    union_area = area_boxA + area_boxB - intersection_area
    iou = float(intersection_area) / float(union_area)
    '''

    ##################################################
    # End of TODO                                    #
    ##################################################
    return iou

def calculate_iou(proposal_boxes, gt_boxes):
    """
    Returns the bounding boxes that have Intersection over Union (IOU) > 0.5 with the ground truth boxes
    :param proposal_boxes: List of proposed bounding boxes(x1,y1,x2,y2) where (x1,y1) indicates the top left corner
    and (x2,y2) indicates the bottom right corner of the proposed bounding box
    :param gt_boxes: List of ground truth boxes(x1,y1,x2,y2) where (x1,y1) indicates the top left corner and (x2,y2)
    indicates the bottom right corner of the ground truth box
    :return iou_qualified_boxes: List of all proposed bounding boxes that have IOU > 0.5 with any of the ground
    truth boxes
    :return final_boxes: List of the best proposed bounding box with each of the ground truth box (if available)
    """
    iou_qualified_boxes = []
    final_boxes = []
    for gt_box in gt_boxes:
        best_box_iou = 0
        best_box = 0
        #area_gt_box = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        for prop_box in proposal_boxes:
            #area_prop_box = (prop_box[2] - prop_box[0] + 1) * (prop_box[3] - prop_box[1] + 1)
            #intersection_area = get_intersection_area(prop_box, gt_box)
            #union_area = area_prop_box + area_gt_box - intersection_area
            #iou = float(intersection_area) / float(union_area)
            iou = bb_intersection_over_union(gt_box, prop_box)
            if iou > 0.5:
                iou_qualified_boxes.append(prop_box)
                if iou > best_box_iou:
                    best_box_iou = iou
                    best_box = prop_box
        if best_box_iou != 0:
            final_boxes.append(best_box)
    return iou_qualified_boxes, final_boxes

def selective_search(img, strategy):
    """
    @brief Selective search with different strategies
    @param img The input image
    @param strategy The strategy selected ['color', 'all']
    @retval bboxes Bounding boxes
    """
    print("Strategy : ", strategy)
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    if strategy == 'color':
        strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        ss.addStrategy(strategy_color)
    else:
        strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        strategy_fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
        strategy_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
        strategy_texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
        strategy_multiple = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(strategy_color, 
                                                                           strategy_fill, strategy_size, strategy_texture)
        ss.addStrategy(strategy_multiple) 
    '''
    sigma: smoothness of the boundary line (small value for complex boundaries, higher numbers to smooth boundaries)
    k: Maybe, (division if the value is less many small area, and the less large area large?) how much to integrate the candidate region
    min_size: minimum size of the area (perhaps region number of pixels)
    '''
    gs = cv2.ximgproc.segmentation.createGraphSegmentation(sigma=0.8, k=200)
    ##################################################
    # TODO: For this part, please set the K as 200,  #
    #       sigma as 0.8 for the graph segmentation. #
    #       Use gs as the graph segmentation for ss  #
    #       to process after strategies are set.     #
    ##################################################

    # set input image on which we will run segmentation
    #ss.setBaseImage(img)
    # Convert image from BGR (default color in OpenCV) to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ss.addImage(rgb_img)

    ss.addGraphSegmentation(gs)
    
    '''
    i) add an image in the RGB format
    (ii) add a graph segmentation method.
    (iii) add the required strategies.
    '''


    ##################################################
    # End of TODO                                    #
    ##################################################
    # run selective search segmentation on input image
    bboxes = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(bboxes)))

    # Convert (x,y,w,h) parameters for [the top 100] all the proposal boxes returned from ss.process() command into
    # (x, y, x+w, y+h) parameters to be consistent with the xml tags of the ground truth boxes where
    # (x,y) indicates the top left corner and (x+w,y+h) indicates the bottom right corner of bounding box
    xyxy_bboxes = []

    for box in bboxes:
        x, y, w, h = box
        xyxy_bboxes.append([x, y, x+w, y + h])

    return xyxy_bboxes

def parse_annotation(anno_path):
    """
    @brief Parse annotation files for ground truth bounding boxes
    @param anno_path Path to the file
    """
    tree = ET.parse(anno_path)
    root = tree.getroot()
    gt_bboxes = []
    for child in root:
        if child.tag == 'object':
            for grandchild in child:
                if grandchild.tag == "bndbox":
                    x0 = int(grandchild.find('xmin').text)
                    x1 = int(grandchild.find('xmax').text)
                    y0 = int(grandchild.find('ymin').text)
                    y1 = int(grandchild.find('ymax').text)
                    gt_bboxes.append([x0, y0, x1, y1])
    return gt_bboxes


'''
You can visualize it by using cv2.rectangle function. 
We provide the function parse_annotation to load the ground truth annotations.
This function returns a list of bounding boxes for the ground truth.
'''
def visualize(img, boxes, color):
    """
    @breif Visualize boxes
    @param img The target image
    @param boxes The box list
    @param color The color
    """
    for box in boxes:
        ##################################################
        # TODO: plot the rectangles with given color in  #
        #       the img for each box.                    #
        ##################################################
        x0, y0, x1, y1 = box
        start_point = (x0,y0)
        end_point = (x1,y1)
        thickness = 1 # 2 px

        cv2.rectangle(img, start_point, end_point, color, thickness)
        cv2.imshow("image", img)
        cv2.waitKey(10)
        cv2.destroyAllWindows()






        ##################################################
        # End of TODO                                    #
        ##################################################
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, default='color')

    args =parser.parse_args()
    img_dir = './HW2_Data/JPEGImages'
    anno_dir = './HW2_Data/Annotations'
    thres = .5

    

    img_list = os.listdir(img_dir)
    num_hit = 0
    num_gt = 0

    for img_path in img_list:
        """
        Load the image file here through cv2.imread
        """
        img_id = img_path[:-4]
        img_name = os.path.join(img_dir, img_path)
        ##################################################
        # TODO: Load the image with OpenCV               #
        ##################################################
        img = cv2.imread(img_name)
        print(img.shape)

        #display
        cv2.imshow(" image ", img); cv2.waitKey(5000)
        cv2.destroyAllWindows()

        





        ##################################################
        # End of TODO                                    #
        ##################################################

        proposals = selective_search(img, args.strategy)
        gt_bboxes = parse_annotation(os.path.join(anno_dir, img_id + ".xml"))
        iou_bboxes = []  # proposals with IoU greater than 0.5

        # Fetch all proposed bounding boxes that have IOU > 0.5 with any of the ground truth boxes and also the bounding box
        # that has the maximum/best overlap for each ground truth box
        iou_bboxes, final_boxes = calculate_iou(proposals, gt_bboxes)
        print("Number of Qualified Boxes with IOU > 0.5 = ", len(iou_bboxes))
        print("Qualified Boxes = ", iou_bboxes)
        ##################################################
        # TODO: For all the gt_bboxes in each image,     #
        #       please calculate the recall of the       #
        #       gt_bboxes according to the document.     #
        #       Store the bboxes with IoU >= 0.5         #
        #       If there is more than one proposal has   #
        #       IoU >= 0.5 with a same groundtruth bbox, #
        #       store the one with biggest IoU.          #
        ##################################################
        print("Number of final boxes = ", len(final_boxes))
        print("Final boxes = ", final_boxes)

        # Recall is calculated as the fraction of ground truth boxes that overlap with at least one proposal box with
        # Intersection over Union (IoU) > 0.5
        recall = len(final_boxes) / len(gt_bboxes)
        print("Recall = ", recall)





        

        ##################################################
        # End of TODO                                    #
        ##################################################
        
        vis_img = img.copy()
        output_img_iou_qualified = img.copy()
        output_img_final = img.copy()
        output_img_proposals = img.copy()

        vis_img = visualize(vis_img, gt_bboxes, (255, 0, 0)) #RGB standard but CV2 will take as BGR; BLUE
        vis_img = visualize(vis_img, iou_bboxes, (0, 0, 255)) # RED [iou > 0.5]

        
        proposals_img = img.copy()
        proposals_img = visualize(proposals_img, gt_bboxes, (255, 0, 0))
        proposals_img = visualize(proposals_img, proposals, (0, 0, 255))
        
        ##################################################
        # TODO: (optional) You may use cv2 to visualize  #
        #       or save the image for report.            #
        ##################################################

        # Draw bounding boxes for proposals
        for i in range(0, len(proposals)):
            top_x, top_y, bottom_x, bottom_y = proposals[i]
            cv2.rectangle(output_img_proposals, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA) #GREEN
        for i in range(0, len(gt_bboxes)):
            top_x, top_y, bottom_x, bottom_y = gt_bboxes[i]
            cv2.rectangle(output_img_proposals, (top_x, top_y), (bottom_x, bottom_y), (255, 0, 0), 1, cv2.LINE_AA) # BLUE
        cv2.imshow("Output_Proposals", output_img_proposals)
        iou_output_img = "./Results/" + str(img_id) + "_proposals.png"
        cv2.imwrite(iou_output_img, output_img_proposals)
        cv2.waitKey(10)
        cv2.destroyAllWindows()

        #output_img_iou_qualified = img.copy()
        # Draw bounding boxes for iou_qualified_boxes
        for i in range(0, len(iou_bboxes)):
            top_x, top_y, bottom_x, bottom_y = iou_bboxes[i]
            cv2.rectangle(output_img_iou_qualified, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA) #GREEN
        for i in range(0, len(gt_bboxes)):
            top_x, top_y, bottom_x, bottom_y = gt_bboxes[i]
            cv2.rectangle(output_img_iou_qualified, (top_x, top_y), (bottom_x, bottom_y), (255, 0, 0), 2, cv2.LINE_AA) # BLUE
        cv2.imshow("Output_IOU_Qualified_Proposals", output_img_iou_qualified)
        iou_output_img = "./Results/" + str(img_id) + "_IOU_Qualifiedl.png"
        cv2.imwrite(iou_output_img, output_img_iou_qualified)
        cv2.waitKey()
        cv2.destroyAllWindows()


        #output_img_final = img.copy()
        # Draw bounding boxes for final_boxes
        for i in range(0, len(final_boxes)):
            top_x, top_y, bottom_x, bottom_y = final_boxes[i]
            cv2.rectangle(output_img_final, (top_x, top_y), (bottom_x, bottom_y), (0, 255, 0), 1, cv2.LINE_AA) #GREEN
        for i in range(0, len(gt_bboxes)):
            top_x, top_y, bottom_x, bottom_y = gt_bboxes[i]
            cv2.rectangle(output_img_final, (top_x, top_y), (bottom_x, bottom_y), (255, 0, 0), 2, cv2.LINE_AA) # BLUE
        cv2.imshow("Output_Final_Boxes", output_img_final)
        output_img = "./Results/" + str(img_id) + "_img_final.png"
        cv2.imwrite(output_img, output_img_final)
        cv2.waitKey()
        cv2.destroyAllWindows()


        ##################################################
        # End of TODO                                    #
        ##################################################
        


if __name__ == "__main__":
    main()




