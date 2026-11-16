#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import typing as tp
import math


# # 50 images saved from all different categories with size 1024x1024 

# In[212]:


base_path = "WarpDoc/distorted"

subfolders = ["curved", "fold", "incomplete", "perspective", "random", "rotate"]

images_dict = {}

for folder in subfolders:
    folder_path = os.path.join(base_path, folder)
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg'))]
    
    folder_images = []
    counter = 0
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv.imread(image_path)
        resized_img = cv.resize(img, (1024, 1024), interpolation=cv.INTER_AREA)
        if img is not None:
            folder_images.append(resized_img)
        else:
            print(f"{image_file} okunamadı.")
        counter += 1
        if counter == 50:
            break
    
    images_dict[folder] = folder_images

print(list(images_dict.keys()))


# In[3]:


num_folders = len(images_dict)
num_images = 5

plt.figure(figsize=(15, num_folders * 3))

for row, (folder, images) in enumerate(images_dict.items()):
    for col in range(min(num_images, len(images))):
        plt.subplot(num_folders, num_images, row * num_images + col + 1)
        img_rgb = cv.cvtColor(images[col], cv.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"{folder} {col+1}")
        plt.axis('off')

plt.tight_layout()
plt.show()


# # Before start any process we need to convert to grayscale to get better performance from our processes

# In[4]:


gray_dict = {}
for key in images_dict.keys():
    for i, img in enumerate(images_dict[key]):
        if key not in gray_dict.keys():
            gray_dict[key] = []
            gray_dict[key].append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
        else:
            gray_dict[key].append(cv.cvtColor(img, cv.COLOR_BGR2GRAY))


# In[5]:


plt.figure(figsize=(15, num_folders * 3))
for row, (folder, images) in enumerate(gray_dict.items()):
    for col in range(min(num_images, len(images))):
        plt.subplot(num_folders, num_images, row * num_images + col + 1)
        plt.imshow(images[col], cmap='gray')
        plt.title(f"{folder} {col+1}")
        plt.axis('off')

plt.tight_layout()
plt.show()


# # After convert all images to grayscale we make gaussian blur and canny edge detection before start hough transform. The parameters is decided after some experiments

# In[6]:


gray = gray_dict['curved'][0]

blurred = cv.GaussianBlur(gray, (15, 15), 0) 

edges = cv.Canny(blurred, 25, 100)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(gray, cmap="gray")
plt.title("Gray Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap="gray")
plt.title("Edges with Canny")
plt.axis("off")

plt.tight_layout()
plt.show()


# In[7]:


cannyEdges = {}

for key, images in gray_dict.items():
    cannyEdges[key] = []  
    for i, img in enumerate(images):        
        blurred = cv.GaussianBlur(img, (15, 15), 0)
        
        edges = cv.Canny(blurred, 25, 200)
        
        cannyEdges[key].append(edges)


# In[8]:


def custom_hough_transform(edges, 
                           original_image=None, 
                           threshold=100, 
                           draw_lines=True):

    height, width = edges.shape
    
    diag_len = int(math.sqrt(width**2 + height**2))
    
    thetas = np.deg2rad(np.arange(0, 180))
    num_thetas = len(thetas)
    
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
    rho_offset = diag_len  
    
    edge_points = np.argwhere(edges != 0)
    
    for y, x in edge_points:
        for theta_idx in range(num_thetas):
            theta = thetas[theta_idx]
            rho = int(round(x * math.cos(theta) + y * math.sin(theta)))
            rho_idx = rho + rho_offset
            accumulator[rho_idx, theta_idx] += 1
    
    lines = []
    for r_idx in range(accumulator.shape[0]):
        for t_idx in range(accumulator.shape[1]):
            if accumulator[r_idx, t_idx] > threshold:
                rho_val = r_idx - rho_offset
                theta_val = thetas[t_idx]
                lines.append((rho_val, theta_val))
    
    drawn_image = None
    if (original_image is not None) and draw_lines:
        drawn_image = original_image.copy()
        
        for (rho, theta) in lines:
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            
            cv.line(drawn_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    return lines, drawn_image

line, lineImg = custom_hough_transform(cannyEdges['curved'][0],images_dict['curved'][0],105)

plt.figure(figsize=(12, 6))
plt.subplot(1,1,1)
plt.imshow(lineImg)


# # Implementing all images to Hough Transform

# In[9]:


houghTransformImagesDict = {}
linesOfImages = {}
for key in cannyEdges.keys():
    for i, edges in enumerate(cannyEdges[key]):
        print(i)
        lines, lineImg = custom_hough_transform(edges,images_dict[key][i])
        if key not in houghTransformImagesDict.keys():
            houghTransformImagesDict[key] = []
            houghTransformImagesDict[key].append(lineImg)
            linesOfImages[key] = []
            linesOfImages[key].append(lines)
        else:
            houghTransformImagesDict[key].append(lineImg)
            linesOfImages[key].append(lines)


# In[10]:


plt.figure(figsize=(15, num_folders * 3))
for row, (folder, images) in enumerate(houghTransformImagesDict.items()):
    for col in range(min(num_images, len(images))):
        plt.subplot(num_folders, num_images, row * num_images + col + 1)
        plt.imshow(images[col], cmap='gray')
        plt.title(f"{folder} {col+1}")
        plt.axis('off')


# # Ransac functions and implementation of Ransac

# In[11]:


def polar_to_endpoints(rho, theta, scale=1000):
    a = math.cos(theta)
    b = math.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + scale * (-b))
    y1 = int(y0 + scale * (a))
    x2 = int(x0 - scale * (-b))
    y2 = int(y0 - scale * (a))
    return (x1, y1, x2, y2)

def sample_line_points(x1, y1, x2, y2, num_points=50):
    xs = np.linspace(x1, x2, num_points)
    ys = np.linspace(y1, y2, num_points)
    return np.vstack((xs, ys)).T

def ransac_line_fitting(points, iterations=1000, threshold=5, min_inliers=15):
    best_line = None
    best_inliers = []
    n_points = len(points)
    if n_points < 2:
        return None, []
    
    for i in range(iterations):
        sample_indices = np.random.choice(n_points, 2, replace=False)
        p1, p2 = points[sample_indices[0]], points[sample_indices[1]]
        if np.array_equal(p1, p2):
            continue
        x1, y1 = p1
        x2, y2 = p2
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        distances = np.abs(a * points[:, 0] + b * points[:, 1] + c) / math.sqrt(a**2 + b**2)
        inlier_indices = np.where(distances < threshold)[0]
        if len(inlier_indices) > len(best_inliers):
            best_inliers = inlier_indices
            best_line = (a, b, c)
            if len(best_inliers) > min_inliers:
                break

    if best_line is not None and len(best_inliers) > 0:
        inlier_points = points[best_inliers]
        vx, vy, x0, y0 = cv.fitLine(inlier_points, cv.DIST_L2, 0, 0.01, 0.01)
        a = vy[0]
        b = -vx[0]
        c = vx[0] * y0[0] - vy[0] * x0[0]
        best_line = (a, b, c)
    return best_line, best_inliers

def line_intersection(line1, line2):
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None  
    x = (b2 * (-c1) - b1 * (-c2)) / det
    y = (a1 * (-c2) - a2 * (-c1)) / det
    return (int(x), int(y))

def visualize_rectangle(img, lines, corners):
    img_copy = img.copy()
    height, width = img.shape[:2]
    for line in lines:
        a, b, c = line
        if abs(b) > 1e-5:
            x1 = 0
            y1 = int((-c - a * x1) / b)
            x2 = width - 1
            y2 = int((-c - a * x2) / b)
        else:
            x1 = x2 = int(-c / a)
            y1, y2 = 0, height - 1
        cv.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for corner in corners:
        if corner is not None:
            cv.circle(img_copy, corner, 5, (0, 0, 255), -1)
    return img_copy

def get_group_points(lines_group):
    all_points = []
    for (x1, y1, x2, y2) in lines_group:
        pts = sample_line_points(x1, y1, x2, y2, num_points=50)
        all_points.append(pts)
    if all_points:
        return np.concatenate(all_points)
    else:
        return np.array([])


afterRansac = {'curved':[],'fold':[], 'incomplete':[], 'perspective':[], 'random':[], 'rotate':[]}
cornersDict = {'curved':[],'fold':[], 'incomplete':[], 'perspective':[], 'random':[], 'rotate':[]}
for key in linesOfImages.keys():
    for i in range(len(linesOfImages[key])):
        lines = linesOfImages[key][i]
        houghDrawnImage = houghTransformImagesDict[key][i]
        
        endpoints_list = [polar_to_endpoints(rho, theta) for (rho, theta) in lines]
        
        vertical_lines = []
        horizontal_lines = []
        for (x1, y1, x2, y2) in endpoints_list:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 45:  # Yaklaşık yatay
                horizontal_lines.append((x1, y1, x2, y2))
            elif abs(angle) > 45:  # Yaklaşık dikey
                vertical_lines.append((x1, y1, x2, y2))
        
        points_horizontal = get_group_points(horizontal_lines)
        points_vertical = get_group_points(vertical_lines)
        
    
        if len(points_horizontal) > 0:
            median_y = np.median(points_horizontal[:, 1])
            pts_top = points_horizontal[points_horizontal[:, 1] < median_y]
            pts_bottom = points_horizontal[points_horizontal[:, 1] >= median_y]
            best_line_top, _ = ransac_line_fitting(pts_top, iterations=10000, threshold=10, min_inliers=50) if len(pts_top) > 0 else (None, [])
            best_line_bottom, _ = ransac_line_fitting(pts_bottom, iterations=10000, threshold=10, min_inliers=50) if len(pts_bottom) > 0 else (None, [])
        else:
            best_line_top = best_line_bottom = None
        
        if len(points_vertical) > 0:
            median_x = np.median(points_vertical[:, 0])
            pts_left = points_vertical[points_vertical[:, 0] < median_x]
            pts_right = points_vertical[points_vertical[:, 0] >= median_x]
            best_line_left, _ = ransac_line_fitting(pts_left, iterations=10000, threshold=10, min_inliers=50) if len(pts_left) > 0 else (None, [])
            best_line_right, _ = ransac_line_fitting(pts_right, iterations=10000, threshold=10, min_inliers= 50) if len(pts_right) > 0 else (None, [])
        else:
            best_line_left = best_line_right = None
    
        corner1 = line_intersection(best_line_left, best_line_top) if best_line_left and best_line_top else None
        corner2 = line_intersection(best_line_right, best_line_top) if best_line_right and best_line_top else None
        corner3 = line_intersection(best_line_right, best_line_bottom) if best_line_right and best_line_bottom else None
        corner4 = line_intersection(best_line_left, best_line_bottom) if best_line_left and best_line_bottom else None
        
        lines_ransac = [l for l in [best_line_left, best_line_right, best_line_top, best_line_bottom] if l is not None]
        corners = [corner1, corner2, corner3, corner4]
        cornersDict[key].append(corners)

        afterRansac[key].append(visualize_rectangle(images_dict[key][i], lines_ransac, corners))



# In[12]:


plt.figure(figsize=(15, num_folders * 3))
for row, (folder, images) in enumerate(afterRansac.items()):
    for col in range(min(num_images, len(images))):
        plt.subplot(num_folders, num_images, row * num_images + col + 1)
        plt.imshow(images[col], cmap='gray')
        plt.title(f"{folder} {col+1}")
        plt.axis('off')


# # Functions for geometric transformation

# In[13]:


def order_points(pts):
    
    pts = np.array(pts)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    
    return np.array([tl, tr, br, bl], dtype="float32")

def compute_destination_size(pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    return maxWidth, maxHeight


# In[14]:


def compute_homography(src_pts, dst_pts):
    num_points = src_pts.shape[0]
    A = []
    for i in range(num_points):
        x, y = src_pts[i][0], src_pts[i][1]
        xp, yp = dst_pts[i][0], dst_pts[i][1]
        # İki satırlık denklem seti:
        A.append([-x, -y, -1,  0,  0,  0, x*xp, y*xp, xp])
        A.append([ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  
    H = h.reshape(3, 3)
    return H / H[2, 2]


# In[15]:


def warpPerspective(src_img, H, dst_size):
    H_inv = np.linalg.inv(H)
    dst_w, dst_h = dst_size
    warped = np.zeros((dst_h, dst_w, src_img.shape[2]), dtype=src_img.dtype)
    
    for i in range(dst_h):
        for j in range(dst_w):
            dst_pt = np.array([j, i, 1], dtype=np.float32)
            src_pt = H_inv.dot(dst_pt)
            src_pt = src_pt / src_pt[2]
            x, y = src_pt[0], src_pt[1]
            if 0 <= x < src_img.shape[1]-1 and 0 <= y < src_img.shape[0]-1:
                x0 = int(np.floor(x))
                x1 = x0 + 1
                y0 = int(np.floor(y))
                y1 = y0 + 1
                a = x - x0
                b = y - y0
                
                warped[i, j] = (1-a)*(1-b)*src_img[y0, x0] + a*(1-b)*src_img[y0, x1] + (1-a)*b*src_img[y1, x0] + a*b*src_img[y1, x1]
    
    return warped


# # Implementing geometric transformations

# In[16]:


warpedImgDict = {'curved':[],'fold':[], 'incomplete':[], 'perspective':[], 'random':[], 'rotate':[]}
for key in cornersDict.keys():
    for i in range(len(linesOfImages[key])):
        valid = True
        for pt in cornersDict[key][i]:
            if pt is None or len(pt) != 2:
                print('patladı')
                valid = False
                break
        print(i)

        
        if not valid:
            warpedImgDict[key].append(None)
        else:
            src = np.array(cornersDict[key][i], dtype=np.float32)
            if src.shape != (4, 2):
                warpedImgDict[key].append(None)
                print(i)
            else:
                rect = order_points(src)
                
                (tl, tr, br, bl) = rect
                
                widthA = np.linalg.norm(br - bl)
                widthB = np.linalg.norm(tr - tl)
                maxWidth = max(int(widthA), int(widthB))
                
                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxHeight = max(int(heightA), int(heightB))
                
                dst = np.array([
                    [0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]
                ], dtype="float32")
        
                H = compute_homography(src, dst)

                try:
                    warpedImg = warpPerspective(images_dict[key][i], H, (maxWidth, maxHeight))
                    warpedImgDict[key].append(warpedImg)
                except np.linalg.LinAlgError:
                    warpedImgDict[key].append(None)


# In[17]:


n_images = len(warpedImgDict['curved'])
cols = 5  
rows = (n_images + cols - 1) // cols 
valid_imgs = [img for img in warpedImgDict['curved'] if img is not None and img.size != 0]

plt.figure(figsize=(15, 3 * rows))
for i, img in enumerate(valid_imgs):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis("off")
plt.tight_layout()
plt.show()


# In[18]:


n_images = len(warpedImgDict['fold'])
cols = 5  
rows = (n_images + cols - 1) // cols  
valid_imgs = [img for img in warpedImgDict['fold'] if img is not None and img.size != 0]

plt.figure(figsize=(15, 3 * rows))
for i, img in enumerate(valid_imgs):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis("off")
plt.tight_layout()
plt.show()


# In[19]:


n_images = len(warpedImgDict['incomplete'])
cols = 5 
rows = (n_images + cols - 1) // cols  
valid_imgs = [img for img in warpedImgDict['incomplete'] if img is not None and img.size != 0]

plt.figure(figsize=(15, 3 * rows))
for i, img in enumerate(valid_imgs):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis("off")
plt.tight_layout()
plt.show()


# In[20]:


n_images = len(warpedImgDict['perspective'])
cols = 5  
rows = (n_images + cols - 1) // cols  
valid_imgs = [img for img in warpedImgDict['perspective'] if img is not None and img.size != 0]

plt.figure(figsize=(15, 3 * rows))
for i, img in enumerate(valid_imgs):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis("off")
plt.tight_layout()
plt.show()


# In[21]:


n_images = len(warpedImgDict['random'])
cols = 5
rows = (n_images + cols - 1) // cols 
valid_imgs = [img for img in warpedImgDict['random'] if img is not None and img.size != 0]

plt.figure(figsize=(15, 3 * rows))
for i, img in enumerate(valid_imgs):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis("off")
plt.tight_layout()
plt.show()


# In[22]:


n_images = len(warpedImgDict['rotate'])
cols = 5
rows = (n_images + cols - 1) // cols  
valid_imgs = [img for img in warpedImgDict['rotate'] if img is not None and img.size != 0]

plt.figure(figsize=(15, 3 * rows))
for i, img in enumerate(valid_imgs):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.title(i)
    plt.axis("off")
plt.tight_layout()
plt.show()


# In[23]:


from skimage.metrics import structural_similarity as ssim

base_path = "WarpDoc/digital"

digitalDict = {}

for folder in subfolders:
    folder_path = os.path.join(base_path, folder)
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg'))]
    
    folder_images = []
    counter = 0
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = cv.imread(image_path)
        resized_img = cv.resize(img, (1024, 1024), interpolation=cv.INTER_AREA)
        if img is not None:
            folder_images.append(resized_img)
        else:
            print(f"{image_file} okunamadı.")
        counter += 1
        if counter == 50:
            break
    
    digitalDict[folder] = folder_images


# In[24]:


num_folders = len(digitalDict)
num_images = 5

plt.figure(figsize=(15, num_folders * 3))

for row, (folder, images) in enumerate(digitalDict.items()):
    for col in range(min(num_images, len(images))):
        plt.subplot(num_folders, num_images, row * num_images + col + 1)
        img_rgb = cv.cvtColor(images[col], cv.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title(f"{folder} {col+1}")
        plt.axis('off')

plt.tight_layout()
plt.show()


# In[25]:


dictForSSIM: {"curved": 1, "fold": 1, "incomplete": 1, "perspective": 1, "random": 1, "rotate": 1}

def match_size(img, ref):
    if img is None or img.size == 0:
        print("Girdi görüntüsü boş!")
        return None
    h_ref, w_ref = ref.shape[:2]
    return cv.resize(img, (w_ref, h_ref), interpolation=cv.INTER_AREA)


# # Curved

# In[91]:


curvedScoresList = []
for i in range(len(warpedImgDict['curved'])):
    score = 0
    if warpedImgDict['curved'][i] is None:
        print('none')
        pass
    else:
        resizedImg = match_size(warpedImgDict['curved'][int(i)], digitalDict['curved'][int(i)])  
        if resizedImg is None:
            pass
        else:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(resizedImg)
            plt.title('curved' + str(i+1))
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.imshow(digitalDict['curved'][int(i)])
            plt.title('curved' + str(i+1))
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            score,map = ssim(resized_img,digitalDict['curved'][i],full=True, win_size=7, channel_axis=-1)
            print(score)
    curvedScoresList.append(score)


# # Because of some SSIM score is not realistic I will make elemination by hand 
# # [1,2,6,8,10,15,17,18,28,34,36,37,38,44,45,46,50] is the images which give at least meaningful results for curved input

# In[174]:


allScores = {'curved':0,'fold':0,'incomplete':0, 'perspective': 0, 'random': 0, 'rotate': 0}
curvedMeaningfullList =  [1,2,6,8,10,15,17,18,28,34,36,37,38,44,45,46,50]
curvedLen = len(curvedMeaningfullList)
for index in curvedMeaningfullList:
    allScores['curved'] += curvedScoresList[index - 1]

allScores['curved'] /= curvedLen
print('ssım mean: ' ,allScores['curved'])
allScores['curved'] += curvedLen / 50
print('freq:', curvedLen/50)
allScores['curved'] /=2


# # Fold

# In[101]:


foldScoresList = []
for i in range(len(warpedImgDict['fold'])):
    score = 0
    if warpedImgDict['fold'][i] is None:
        pass
    else:
        resizedImg = match_size(warpedImgDict['fold'][int(i)], digitalDict['fold'][int(i)])  
        if resizedImg is None:
            pass
        else:
            plt.figure(figsize=(12, 6))
            # İlk görüntü
            plt.subplot(1, 2, 1)
            plt.imshow(resizedImg)
            plt.title('fold' + str(i+1))
            plt.axis("off")
            
            # İkinci görüntü
            plt.subplot(1, 2, 2)
            plt.imshow(digitalDict['fold'][int(i)])
            plt.title('fold' + str(i+1))
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            score,map = ssim(resized_img,digitalDict['fold'][i],full=True, win_size=7, channel_axis=-1)
            print(score)
    foldScoresList.append(score)
#[1,3,7,8,15,16,18,21,29,32,33,35,42,43,44,47,50]


# # [1,3,7,8,15,16,18,21,29,33,35,39,42,43,45,50] is the inputs which is meaningfull for fold inputs

# In[176]:


foldMeaningfullList = [1,3,7,8,15,16,18,21,29,33,35,39,42,43,45,50]
foldLen = len(foldMeaningfullList)
for index in foldMeaningfullList:
    allScores['fold'] += foldScoresList[index -1]

allScores['fold'] /= foldLen
print('ssım mean:',allScores['fold'])
allScores['fold'] += foldLen /50
print('freq:',foldLen /50)
allScores['fold'] /=2


# # Incomplete

# In[103]:


incompleteScoresList = []
for i in range(len(warpedImgDict['incomplete'])):
    score = 0
    if warpedImgDict['incomplete'][i] is None:
        print('none')
        pass
    else:
        resizedImg = match_size(warpedImgDict['incomplete'][int(i)], digitalDict['incomplete'][int(i)])  
        if resizedImg is None:
            pass
        else:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(resizedImg)
            plt.title('incomplete' + str(i+1))
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.imshow(digitalDict['incomplete'][int(i)])
            plt.title('incomplete' + str(i+1))
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            score,map = ssim(resized_img,digitalDict['incomplete'][i],full=True, win_size=7, channel_axis=-1)
            print(score)
    incompleteScoresList.append(score)


# # [1,2,3,7,8,14,15,17,21,23,24,25,26,27,28,29,36,37,41,42,43,44,45] s the inputs which is meaningfull for incomplete inputs¶

# In[178]:


incompleteMeaningfullList = [1,2,3,7,8,14,15,17,21,23,24,25,26,27,28,29,36,37,41,42,43,44,45]
incompleteLen = len(incompleteMeaningfullList)
for index in incompleteMeaningfullList:
    allScores['incomplete'] += incompleteScoresList[index - 1]


allScores['incomplete'] /= incompleteLen
print('ssım mean:',allScores['incomplete'])
allScores['incomplete'] += incompleteLen / 50
print('freq:',incompleteLen/50 )
allScores['incomplete'] /=2


# # Perspective

# In[105]:


perspectiveScoresList = []
for i in range(len(warpedImgDict['perspective'])):
    score = 0
    if warpedImgDict['perspective'][i] is None:
        pass
    else:
        resizedImg = match_size(warpedImgDict['perspective'][int(i)], digitalDict['perspective'][int(i)])  
        if resizedImg is None:
            pass
        else:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(resizedImg)
            plt.title('perspective' + str(i+1))
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.imshow(digitalDict['perspective'][int(i)])
            plt.title('perspective' + str(i+1))
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            score,map = ssim(resized_img,digitalDict['perspective'][i],full=True, win_size=7, channel_axis=-1)
            print(score)
    perspectiveScoresList.append(score)


# # [1,3,4,5,7,9,10,11,15,16,20,21,25,27,30,31,34,36,35,36,37,38,39,41,43,44,45,46,49,50] perspective

# In[180]:


perspectiveMeaningfullList = [1,3,4,5,7,9,11,15,20,23,24,21,25,27,30,33,36,35,37,38,39,41,44,45,46,49,50]
perspectiveLen = len(perspectiveMeaningfullList)
for index in perspectiveMeaningfullList:
    allScores['perspective'] += perspectiveScoresList[index-1]

allScores['perspective'] /= perspectiveLen
print('ssım:',allScores['perspective'])
allScores['perspective'] += perspectiveLen /50
print('freq:',perspectiveLen/50)
allScores['perspective'] /=2


# # Random

# In[107]:


randomScoresList = []
for i in range(len(warpedImgDict['random'])):
    score = 0
    if warpedImgDict['random'][i] is None:
        pass
    else:
        resizedImg = match_size(warpedImgDict['random'][int(i)], digitalDict['random'][int(i)])  
        if resizedImg is None:
            pass
        else:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(resizedImg)
            plt.title('random' + str(i+1))
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.imshow(digitalDict['random'][int(i)])
            plt.title('random' + str(i+1))
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            score,map = ssim(resized_img,digitalDict['random'][i],full=True, win_size=7, channel_axis=-1)
            print(score)
    randomScoresList.append(score)


# # [12,14,17,20,23,23,27,34,49] for random

# In[182]:


randomMeaningfullList = [12,14,17,20,23,23,27,34,49]
randomLen = len(randomMeaningfullList)
for index in randomMeaningfullList:
    allScores['random'] += randomScoresList[index-1]


allScores['random'] /= randomLen
print('ssım:',allScores['random'])
allScores['random'] += randomLen /50
print('freq:',randomLen/50)
allScores['random'] /= 2


# # Rotate

# In[109]:


rotateScoresList = []
for i in range(len(warpedImgDict['rotate'])):
    score = 0
    if warpedImgDict['rotate'][i] is None:
        pass
    else:
        resizedImg = match_size(warpedImgDict['rotate'][int(i)], digitalDict['rotate'][int(i)])  
        if resizedImg is None:
            pass
        else:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(resizedImg)
            plt.title('rotate' + str(i+1))
            plt.axis("off")
            
            plt.subplot(1, 2, 2)
            plt.imshow(digitalDict['rotate'][int(i)])
            plt.title('rotate' + str(i+1))
            plt.axis("off")
            
            plt.tight_layout()
            plt.show()
            score,map = ssim(resized_img,digitalDict['rotate'][i],full=True, win_size=7, channel_axis=-1)
            print(score)
    rotateScoresList.append(score)


# # [1,2,7,8,9,12,15,20,21,23,25,27,28,30,33,37,40,41,45] for rotate

# In[184]:


rotateMeaningfullList = [1,2,7,8,9,12,15,20,21,23,25,27,28,30,33,37,40,41,45]
rotateLen = len(rotateMeaningfullList)
for index in rotateMeaningfullList:
    allScores['rotate'] += rotateScoresList[index-1]

allScores['rotate'] /= rotateLen
print('ssım:',allScores['rotate'])
allScores['rotate'] += rotateLen / 50
print('freq:',rotateLen/50)
allScores['rotate'] /= 2


# In[170]:


print(allScores)


# new_width,new_height = 1024,1024
# cv.imwrite('curvedProper.jpg', warpedImgDict['curved'][0])
# cv.imwrite('curvedBad.jpg',cv.resize(warpedImgDict['curved'][3], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('foldProper.jpg', cv.resize(warpedImgDict['fold'][15], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('foldBad.jpg', cv.resize(warpedImgDict['fold'][19], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('incompleteProper.jpg', cv.resize(warpedImgDict['incomplete'][6], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('incompleteBad.jpg', cv.resize(warpedImgDict['incomplete'][9], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('perspectiveProper.jpg', cv.resize(warpedImgDict['perspective'][0], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('perspectiveBad.jpg', cv.resize(warpedImgDict['perspective'][7], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('randomProper.jpg', cv.resize(warpedImgDict['random'][10], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('randomBad.jpg', cv.resize(warpedImgDict['random'][17], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('rotateProper.jpg', cv.resize(warpedImgDict['rotate'][27], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('rotateBad.jpg', cv.resize(warpedImgDict['rotate'][40], (new_width, new_height), interpolation=cv.INTER_AREA))
# 
# 
# cv.imwrite('DigitalCurvedProper.jpg', cv.resize(digitalDict['curved'][0], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalCurvedBad.jpg', cv.resize(digitalDict['curved'][3], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalFoldProper.jpg', cv.resize(digitalDict['fold'][15], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalfoldBad.jpg', cv.resize(digitalDict['fold'][19], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalincompleteProper.jpg', cv.resize(digitalDict['incomplete'][6], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalincompleteBad.jpg', cv.resize(digitalDict['incomplete'][9], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalperspectiveProper.jpg', cv.resize(digitalDict['perspective'][0], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalperspectiveBad.jpg', cv.resize(digitalDict['perspective'][7], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalrandomProper.jpg', cv.resize(digitalDict['random'][10], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalrandomBad.jpg', cv.resize(digitalDict['random'][17], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalrotateProper.jpg', cv.resize(digitalDict['rotate'][27], (new_width, new_height), interpolation=cv.INTER_AREA))
# cv.imwrite('DigitalrotateBad.jpg', cv.resize(digitalDict['rotate'][40], (new_width, new_height), interpolation=cv.INTER_AREA))
