import cv2
import matplotlib.pyplot as plt
import jsonlines
from PIL import Image

# Load the JSON data
file_path = "output/reclip_test.txt"

for instance in jsonlines.Reader(open(file_path)):
    # Load the image 
    plt.imshow(Image.open("reclip_data/images/" + instance["file_name"]))
    ax = plt.gca()
    
    # Draw ground truth bounding boxes
    for j in instance["gold_index"]:
        gt = instance["bboxes"][j]
        ax.add_patch(plt.Rectangle((gt[0], gt[1]), gt[2] - gt[0], gt[3] - gt[1], color = "blue", fill = False, linewidth = 1))
        plt.text(gt[0], gt[1], 'GT', color='white', bbox={'facecolor':'blue', 'alpha':0.5})

    # Draw predicted bounding box
    pred = instance["box"]
    ax.add_patch(plt.Rectangle((pred[0], pred[1]), pred[2], pred[3], color = "green", fill = False, linewidth = 1))

    # Set image title and adjust visualization settings
    plt.text(pred[0], pred[1], 'PRED', color='white', bbox={'facecolor':'green', 'alpha':0.5})
    plt.title(instance["text"], y = -0.15)
    plt.xticks([])
    plt.yticks([])

    # Show the image with annotations
    plt.show()