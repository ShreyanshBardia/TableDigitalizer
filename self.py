from math import ceil
import cv2, csv, glob, os, easyocr
import numpy as np


def preprocess_invert_image(img:str):
  img = cv2.imread(img)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  img_height, img_width = img.shape
  counts,bins = np.histogram(img,bins=20) #need to experiment with 10 also
  counts_perc = counts/counts.sum()
  cumsum = counts.cumsum()/counts.sum()
  argmax = counts_perc.argmax()
  perc_left = 1 - counts_perc[argmax]
  if cumsum[argmax-1]/perc_left>=0.50:
    threshold = bins[argmax - 2]
    inv_req = True
  else:
    threshold = bins[argmax + 2 ] 
    inv_req = False
  # print(f"{inv_req=}")
  thresh, img_bin = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

  img_bin_inv = img_bin
  if inv_req:
    img_bin_inv = 255 - img_bin
  # cv2_imshow(img_bin_inv)
  return img,255-img_bin_inv,img_bin_inv,False
  



def self_unbordered(img_path):

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
    widths = [boundingBoxes[i][2] for i in range(len(boundingBoxes))]
    mean = np.mean(heights)
    box = []
    median_height = np.median(heights)
    median_width = np.median(widths)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.9*img_width and h < 0.9*img_height):
            if w<median_width*0.5 or h<median_height*0.5:
                continue
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])


    cv2.imwrite("static/files/self.png",image)

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





def self_bordered(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_height, img_width = img.shape
    img_bin = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    img_bin = 255 - img_bin
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
    widths = [boundingBoxes[i][2] for i in range(len(boundingBoxes))]
    median_height = np.median(heights)
    median_width = np.median(widths)
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w < 0.7*img_width and h < 0.7*img_height):
            if w<median_width*0.4 or h<median_height*0.4:
                continue
            image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])
    
    mean = np.mean(heights)

    row = []
    column = []
    j = 0
    for i in range(len(box)):
        column.sort(key=lambda x: (x[1],x[0]))
        x,y,w,h = box[i]
        # if w<median_width*0.4 or h<median_height*0.4:
        #     continue
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
                if (i == len(box) - 1):
                    row.append(column)

    cv2.imwrite("static/files/self.png",image)




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






def recognize_rows(img):
  
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    # img_bin_inv = 255 - img_bin
    img,img_bin,img_bin_inv,_ = preprocess_invert_image(img)
    img_height, img_width = (img).shape
    if (img>180).sum()/(img_height* img_width) >= 0.5:
      thresh, img_bin = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
      img_bin_inv = 255- img_bin
    kernel_len_ver = max(10,img_height // 50)
    kernel_len_hor = max(10, img_width // 50)
    # Defining a vertical kernel to detect all vertical lines of image
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len_ver)) #shape (kernel_len, 1) inverted! xD
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len_hor, 1)) #shape (1,kernel_ken) xD
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin_inv, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=4)
    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin_inv, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=4)
    
    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)

    # Eroding and thesholding the image
    img_vh = cv2.dilate(img_vh, kernel, iterations=5)
    thresh, img_vh = (cv2.threshold(img_vh, 50, 255, cv2.THRESH_BINARY ))

    bitor = cv2.bitwise_or(img_bin, img_vh)

    img_median = cv2.medianBlur(bitor, 3)

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width*2, 1)) #shape (kernel_len, 1) inverted! xD
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)

    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0, horizontal_lines, 1, 0.0)

    # # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)

    thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY )
    bitxor = cv2.bitwise_xor(img, img_vh)
    bitnot = cv2.bitwise_not(bitxor)

    # Detect contours for following box detection
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


    # # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    # Get mean of heights
    mean = np.mean(heights)

    # Create list box to store all boxes in
    box = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        box.append([x, y, w, h])

    # Improving the height of rows
    box_new = box[:]
    running_avg = 0
    for id in range(len(box)-1):
      b1,b2 = box[id],box[id+1]
      avg =( ( b2[1] - b1[1] - b1[3] )/2)
      b1[3]+=int(avg*3/4)
      b2[1]-=ceil(avg*5/4)
      b2[3]+=ceil(avg)
      if id == 0:
        b1[1]-=ceil(avg*5/4)
        if b1[1]<0:
          box_new.remove(b1)
        b1[3]+=ceil(avg)
      if id == len(box)-2:
        b2[3]+=int(avg*3/4)


    for b in box_new:
      x, y, w, h = b
      image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    

    return box_new,img_median,img




def recognize_cols(img_median,img):
    img_height, img_width = img_median.shape


    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, img_height*2)) #shape (kernel_len, 1) inverted! xD
    vertical_lines = cv2.erode(img_median, ver_kernel, iterations=1)

    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_width*2, 1)) #shape (kernel_len, 1) inverted! xD
    horizontal_lines = cv2.erode(img_median, hor_kernel, iterations=1)
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 1, horizontal_lines, 0, 0.0)

    # # Eroding and thesholding the image
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


    # # Sort all the contours by left to right.
    contours, boundingBoxes = sort_contours(contours, method="left-to-right")

    box = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        box.append([x, y, w, h])

    img_width = img.shape[1]

    for id in range(len(box)-1):
      b1,b2 = box[id],box[id+1]
      avg =( ( b2[0] - b1[0] - b1[2] )/2)
      
      b1[2]+=int(avg*3/4)
      b2[0]-=ceil(avg*5/4)
      b2[2]+=ceil(avg)
      
      if id == 0:
        b1[2]+=b1[0]
        b1[0]=0

      if id==len(box)-2:
        b2[2]=img_width-b2[0]
        assert b2[2]+b2[0] == img_width
    # blank = np.ones_like(img)*255
    for b in range(len(box)):
      x, y, w, h = box[b]
      if b!=0 or b!=len(box)-1:
        image = cv2.line(img, (x+w,0), (x+w,img.shape[0]), (0, 255, 0), 2)
        # blank = cv2.line(blank, (x+w,0), (x+w,img.shape[0]), (0, 255, 0), 2)
    try:
      # cv2_imshow(image)
      # cv2_imshow(blank)
      pass
    except:
      # cv2_imshow(img)
      # cv2_imshow(blank)
      pass
    return box if len(box)>1 else -1

    

        



def recognize_structure(read):
  row_boxes,image,img=recognize_rows((read))
  col_boxes = []
  index=0
  for x,y,w,h in row_boxes: 
    i = (cv2.imread(read)[y:y+h,x:x+w])
    try:
      col_box = recognize_cols(image[y:y+h,x:x+w],img[y:y+h,x:x+w])
      col_boxes.append(col_box)

    except:
      col_boxes.append([])

  assert len(row_boxes)==len(col_boxes)
  return row_boxes,col_boxes



reader = easyocr.Reader(lang_list=["en"])



def img_to_table(img_path:str):
  img = cv2.imread(img_path)
  reader = easyocr.Reader(lang_list=["en"])
  row_boxes,col_boxes = recognize_structure(img_path)
  print(f"{col_boxes=}")
  col_boxes = sanitize(img_path, row_boxes, col_boxes)
  print("Started in img_to_table \n\n\\n\n\n\n\n")
  img_bw,img_bin,img_bin_inv,use_tess = preprocess_invert_image(img_path)
  data= []
  max_=0
  for idx,row in enumerate(row_boxes):
    x,y,w,h = row
    data.append([])
    try:
      if x<0 or y<0 or w<0 or h<0:
        continue
      if col_boxes[idx]!=-1:
        for col in col_boxes[idx]:
          x,_,w,_ = col
          if x<0 or w<0:
            continue
          if use_tess:
            text = (pytesseract.image_to_string(img_bw[ y:y+h, x:x+w]))
          else:
            text=""
            for detected in (reader.readtext(img_bin[ y:y+h, x:x+w])):
              text += detected[1] 
          data[-1].append(text)
      else:
        if use_tess:
          data[-1].append(pytesseract.image_to_string(img_bw[ y:y+h, x:x+w]))
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
  for id,row in enumerate(row_boxes):
    x,y,w,h = row
    img_row = img[y:y+h,x:x+w]
    if col_boxes[id]!=-1:
      for box_id in range(len(col_boxes[id])):
        x, y, w, h = col_boxes[id][box_id]
        image = cv2.line(img_row, (x+w,0), (x+w,img_row.shape[0]), (0, 0, 0), 2)
        image = cv2.line(img_row,(0,0), (x+w,0), (0, 0, 0), 3)
    else:
      image = cv2.line(img_row,(0,0), (x+w,0), (0, 0, 0), 3)
    if id==0:
      z=image
    else:
      z=np.vstack((z,image))
  cv2.imwrite("static/files/self.png",z)
  return data,row_boxes,col_boxes




def html_output(img_path:str):
  # cv2_imshow(cv2.imread(img_path))
  data,row_boxes,col_boxes = img_to_table(img_path)
  print(col_boxes)
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




def sanitize(img_path,row_boxes,col_boxes):
  img=cv2.imread(img_path)
  l=[]
  for row in row_boxes:
    x,y,w,h = row
    try:
      l.append(sorted(reader.readtext(img[y:y+h,x:x+w]),key = lambda x:x[0][1][0]))
    except:
      pass
  sanity=[]
  index=0
  for col,easy in zip(col_boxes,l):
    con=True
    if not easy:
      sanity.append(col)
      index+=1
      con=False
      continue
    if con:
      for x in easy:
        if "   " in x[1]:
          con=False
          sanity.append(col)
          continue
    if col!=-1 and con:
        length = len(col)
        idx_easy,idx_col =0,0
        output=[]
        while idx_col<length:
          x_col,_,w_col,_=col[idx_col]
          
          x,y,w,h = col[idx_col]
          if idx_easy<len(easy):
            easy_border = easy[idx_easy][0][1][0]
            easy_start = easy[idx_easy][0][0][0]
            if x_col+w_col+4<easy_border and x_col+w_col>easy_start:
              idx_col+=1
            else:
              if not output:
                output.append([0,y,x+w,h])
              else:
                x_1,y_1,w_1,h_1 = output[-1]
                end = x+w
                start = x_1+w_1
                if start!=end:
                  output.append([start,y,end-start,h])
              idx_col+=1
              if easy_border<x_col+w_col:
                idx_easy+=1
          else:
            if not output:
              output = col
            else:
              x_1,y_1,w_1,h_1 = output[-1]
              end = x+w
              start = x_1+w_1
              if start!=end:
                output.append([start,y,end-start,h])
              idx_col+=1
        sanity.append(output)
    elif col==-1:
      sanity.append(-1)
  return sanity
    
