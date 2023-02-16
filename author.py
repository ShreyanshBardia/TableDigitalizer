import cv2
import numpy as np
from self import preprocess_invert_image
import easyocr




reader = easyocr.Reader(lang_list=["en"])

def author_partially_bordered(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape 
    thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    img_bin_inv = 255 - img_bin
    kernel_len_ver = max(10,img_height // 50)
    kernel_len_hor = max(10, img_width // 50)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver)) 
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1)) 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.dilate(img_vh, kernel, iterations=5)
    thresh, img_vh = (cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY ))
    bitor = cv2.bitwise_or(img_bin, img_vh)
    img_median = cv2.medianBlur(bitor, 3)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, img_height*2)) 
    vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width*2, 1)) 
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY )
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    def sort_contours(cnts, method="left-to-right"):
        
        reverse = False
        i = 0
        
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        
        
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        
        
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        
        return (cnts, boundingBoxes)
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    
    box = []
    
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        if (w < 0.9*img_width and h < 0.9*img_height):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])

    cv2.imwrite('static/files/author.png',image)





def author_unbordered(img_path):

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    img_median = cv2.medianBlur(img_bin, 3)
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, img_height*2)) #shape (kernel_len, 1) inverted! xD
    vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width*2, 9)) #shape (kernel_len, 1) inverted! xD
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY )
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9*img_width and h < 0.9*img_height):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])


    cv2.imwrite("static/files/author.png",image)

    row = []
    column = []
    j = 0
    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]

        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]

                if (i == len(box) - 1):
                    row.append(column)

            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])


    # print(list(map(len,row)))
    img_bw,img_bin,img_bin_inv,use_tess = preprocess_invert_image(img_path)
    data= []
    max_=0

    for r in (row):
      data.append([])
      try:
        for col in r:
          x,y,w,h = col
          if use_tess:
            text = (pytesseract.image_to_string(img_bw[ y:y+h, x:x+w]))
          else:
            text=""
            for detected in (reader.readtext(img_bin[ y:y+h, x:x+w])):
              text += detected[1] 
          data[-1].append(text)
      except Exception as e:
        data.pop()
        print(e)
    
      if max_<=len(data[-1]):
        max_ = len(data[-1])
      else:
        data[-1] = data[-1]+[""]*(max_-len(data[-1]))
    # print(data)
    table=""
    for row in data:
      table+="<tr>\n"
      for idx,col in enumerate(row):
        if idx==0 and (not col.strip()):
          pass
        else:
          col = col.replace("\n","").replace("\x0c","").replace("!","").replace("|","").title()
          table+=f"<td> {col} </td>\n"
      table+="</tr>\n"
    return (f"<table>\n{table}</table>")






def author_bordered(img):
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    img_bin = 255 - img_bin
    # plotting = plt.imshow(img_bin, cmap='gray')
    kernel_len_ver = img_height // 50
    kernel_len_hor = img_width // 50
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver))  
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1))  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY )
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    def sort_contours(cnts, method="left-to-right"):        
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i], reverse=reverse))
        return (cnts, boundingBoxes)
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    widths = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]
    mean_w = np.mean(widths)
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9*img_width and h < 0.9*img_height):
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])
    
    mean = np.mean(heights)
    row = []
    column = []
    j = 0
    for i in range(len(box)):
        if (i == 0):
            column.append(box[i])
            previous = box[i]

        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]

                if (i == len(box) - 1):
                    row.append(column)

            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    cv2.imwrite("static/files/author.png",image)
    return sorted(box,key=lambda x: (x[1],x[0])),row,column