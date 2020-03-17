import pandas as pd
import numpy as np
import sys
import json

from detectron2.structures import BoxMode

try:
	CSVFILE = sys.argv[1]
	OUTFILE = sys.argv[2]
except:
	print("Exiting. Usage: python convert_csv_to_detectron.py [csvfile] [jsonfile]")


df = pd.read_csv(CSVFILE)

unique_classes = df['class'].unique()
N_classes = len(unique_classes)
class_to_int = dict(zip(unique_classes, np.arange(N_classes)))

df['class_int'] = df['class'].apply(lambda x: class_to_int[x])

df_groups = df.groupby('filename')

data = []
for idx, g in enumerate(df_groups): #loop over every image
	assert(len(g[1]['height'].unique())==1)
	assert(len(g[1]['width'].unique())==1)

	g1 = g[1].transpose().to_dict()
	g1_keys = list(g1.keys())

	filename = g[0]
	height = int(g[1].iloc[0].height)
	width = int(g[1].iloc[0].width)

	if height==0 or width==0:
		continue #temporary fix
		p = subprocess.Popen(["identify", filename], stdout=subprocess.PIPE)
		out = p.stdout.readline()

		hw = out.split()[2]
		hw = output.decode('utf-8').split()[2]
		if hw.find('x')==-1:
			raise ValueError('Error in conversion. Size not found')
		h, w = hw.split('x')
		h, w = int(h), int(w)

	d = {
		'image_id': idx,
		'file_name': filename,
		'height': height,
		'width': width,
		'annotations': []
	}

	for k in g1_keys: #loop over annotations
		ann = {'bbox': (int(g1[k]['xmin']), int(g1[k]['ymin']), int(g1[k]['xmax']), int(g1[k]['ymax'])),
			   'bbox_mode': BoxMode.XYXY_ABS, 
			   'category_id': int(g1[k]['class_int'])}

		d['annotations'].append(ann)

	data.append(d)

json.dump(data, open(OUTFILE, 'w'))
