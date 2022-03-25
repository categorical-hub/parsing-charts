from pdf2image import convert_from_path
import cv2
import pytesseract
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
kernel_size = (39, 39)

images = convert_from_path('chart.pdf')
items = []
full_items = []


for i in range(len(images)):
    img = images[i]

    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)

    im2 = np.asarray(img.copy())

    items = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped = im2[y:y + h, x:x + w]
        text = pytesseract.image_to_string(cropped)
        items.append((text, (x, y, w, h)))

    items.reverse()
    items_print_order = items.copy()
    # Routine - remove headers and footers
    items.sort(key=lambda temp: temp[1][1])
    top = bottom = 0
    headers_line = [items[-1][1][1], items[0][1][1]]

    items_print_order = [x for x in items_print_order if x[1][1] not in headers_line]

    cv2.imwrite('/home/sagi/test_temp{}.jpg'.format(i), im2)

    full_items.extend(items_print_order)
    items = []



graph = []
titles = ['medical decision making', 'hpi:', 'physical exam:', 'ed course:', 'past medical history:',
          'past surgical history:', 'history of present illness:', 'pmhx:', 'social history:', 'family history:',
          'allergies:']

parent = 'chart'

for it in full_items:
    node = it[0]
    # if found title:
    if node.strip() == '':
        continue
    if len([x for x in titles if x in node.lower()]) > 0:
        graph.append(('chart', node))
        parent = node
    else:
        graph.append((parent, node))
        parent = node

plt.figure(figsize=(58, 58))
cnt = 0
G = nx.DiGraph()
# G.add_edges_from(graph)
for node in graph:
    if node[0] not in G:
        G.add_node(node[0])

for edge in graph:
    G.add_edge(edge[0], edge[1])
    cnt += 1

nt = Network('500px', '500px')
nt.from_nx(G)
nt.show('/home/sagi/test.html')
