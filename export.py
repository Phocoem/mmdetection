import matplotlib.pyplot as plt
import cv2

# load ảnh
input_img = cv2.imread('input.png')
output_img = cv2.imread('result.png')

input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

fig, ax = plt.subplots(figsize=(12, 4))

# tắt trục
ax.axis('off')

# vẽ ảnh input
ax.imshow(input_img, extent=(0, 2, 0, 2))

# vẽ ảnh output
ax.imshow(output_img, extent=(8, 10, 0, 2))

# text pipeline
ax.text(2.5, 1, "Backbone\n(ResNet-50)", fontsize=10)
ax.text(4.5, 1, "FPN", fontsize=10)
ax.text(6.5, 1, "RoIAlign + Heads", fontsize=10)

# arrows
ax.arrow(2, 1, 0.5, 0, head_width=0.1)
ax.arrow(4, 1, 0.5, 0, head_width=0.1)
ax.arrow(6, 1, 0.5, 0, head_width=0.1)
ax.arrow(7.5, 1, 0.5, 0, head_width=0.1)

plt.savefig("pipeline.png", bbox_inches='tight')
plt.show()