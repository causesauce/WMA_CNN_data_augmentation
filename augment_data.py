import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

# %%
# images = []
# for i in range(1, 11):
#     path = f'data/orange/{i}.jpg'
#     im = cv.imread(path, cv.IMREAD_UNCHANGED)
#     im = cv.resize(im, (128, 128), interpolation=cv.INTER_AREA)
#     images.append(im)
#     cv.imwrite(path, im)

# %%
# images = np.array(images)
# for i in images:
#     plt.imshow(i)
#     plt.show()

# %%

fruit = "lemon"
counter = 11
for i in range(1, 11):
    path = f'data/{fruit}/{i}.jpg'
    im = cv.imread(path, cv.IMREAD_UNCHANGED)
    desired_size = max(im.shape[:2])
    old_size = im.shape[:2]

    new_size = old_size
    #im = cv.resize(im, (new_size[0], new_size[1]))
    delta_w = desired_size  - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [255, 255, 255]
    im = cv.copyMakeBorder(im, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)

    im_to_rewrite = cv.resize(im, (128, 128))
    # cv.imshow(f"{counter}", im_modified)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    cv.imwrite(path, im_to_rewrite)

    for j in range(20):
        should_repeat = True
        while should_repeat:
            rotate = random.randint(1, 10)
            scale = random.randint(1, 10)
            translate = random.randint(1, 10)
            im_modified = np.copy(im)

            if translate % 2 == 0:
                height, width = im_modified.shape[:2]
                limit_height, limit_width = height // 4, width // 4
                x_minus = random.randint(0, 1)
                y_minus = random.randint(0, 1)
                x_translate = random.randint(5, limit_width)
                y_translate = random.randint(5, limit_height)
                if x_minus:
                    x_translate = -x_translate
                if y_minus:
                    y_translate = -y_translate

                T = np.float32([[1, 0, x_translate], [0, 1, y_translate]])

                im_modified = cv.warpAffine(im_modified, T, (width, height))

            if rotate % 2 == 0:
                im_modified = cv.rotate(im_modified, random.randint(0, 2))

            if scale % 2 == 0 or scale % 3 == 0:
                rand_scale_coeff = random.randint(30, 90)
                height = int(im_modified.shape[1] * rand_scale_coeff / 100)
                width = int(im_modified.shape[0] * rand_scale_coeff / 100)
                dim = (height, width)

                im_modified = cv.resize(im_modified, (height, width), interpolation=cv.INTER_AREA)

            if translate % 2 == 0 or rotate % 2 == 0 or scale % 2 == 0 or scale % 3 == 0:
                #print(counter)
                im_modified = cv.resize(im_modified, (128, 128))
                #cv.imshow(f"{counter}", im_modified)
                #cv.waitKey(0)
                #cv.destroyAllWindows()
                cv.imwrite(f'data/{fruit}/{counter}.jpg', im_modified)
                should_repeat = False

        counter += 1

# %%
# import cv2 as cv
#
# fruit = "orange"
# for i in range(1, 211):
#     path = f'data/{fruit}/{i}.jpg'
#     im = cv.imread(path, cv.IMREAD_UNCHANGED)
#     print(im.shape)
